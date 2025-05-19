import cv2
import paddle
import numpy as np
from paddle.vision import mobilenet_v3_small
import queue
import threading
import csv
import dataset_utils
from dataset_utils import datasets
from dataset_utils import eval_transforms
import time
import os
from PIL import Image
import config

def preprocess_frame(frame):
    """
    将OpenCV BGR帧转换为模型输入格式
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    processed_tensor = eval_transforms(pil_img)
    if not isinstance(processed_tensor, paddle.Tensor):
        processed_tensor = paddle.to_tensor(processed_tensor)
    return paddle.unsqueeze(processed_tensor, axis=0)

def listen_web_carmer():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("无法打开视频流")
        cap = cv2.VideoCapture(0)  # Try local webcam (index 0)
        if not cap.isOpened():
            print("ERROR: Failed to open local webcam!")
            return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流结束或连接中断")
            break
        if frame_count % frame_time == 0:
            if frame_queue.full():
                with condLock_full:
                    condLock_full.wait_for(lambda: not frame_queue.full())
            frame_queue.put(frame)
            with condLock_empty:
                condLock_empty.notify()
            cv2.imshow("原始视频流", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下Q键，退出捕获")
                break
        frame_count += 1
    cap.release()
    cv2.destroyWindow("原始视频流")
    print("网络摄像头线程结束")


def local_video():
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        print(f"无法打开本地视频: {local_video_path}")
        return    
    frame_count = 0
    print(f"正在播放本地视频: {local_video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束")
            break
        
        if frame_count % frame_time == 0:
            if frame_queue.full():
                with condLock_full:
                    condLock_full.wait_for(lambda: not frame_queue.full())
            frame_queue.put(frame)
            with condLock_empty:
                condLock_empty.notify()
            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下Q键，退出播放")
                break
        frame_count += 1
        time.sleep(0.01)
    
    cap.release()
    cv2.destroyWindow("video")
    print("本地视频线程结束")

def read_labels_to_dict(filepath):
    """读取CSV标签文件并返回字典"""
    result = {}  # 使用局部变量而不是全局变量
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_id = int(row['ClassId'])
                name = row['Name']
                result[class_id] = name
        return result  # 返回结果字典
    except Exception as e:
        print(f"读取标签CSV文件{filepath}时出错: {e}")
        return {}


def get_center_crop(frame, scale=2/3):
    height, width = frame.shape[:2]
    
    center_width = int(width * scale)
    center_height = int(height * scale)
    
    start_x = (width - center_width) // 2
    start_y = (height - center_height) // 2
    
    center_crop = frame[start_y:start_y+center_height, start_x:start_x+center_width]
    return center_crop

def split_into_four_blocks(frame):
    height, width = frame.shape[:2]
    
    half_width = width // 2
    half_height = height // 2
    
    top_left = frame[0:half_height, 0:half_width]
    top_right = frame[0:half_height, half_width:width]
    bottom_left = frame[half_height:height, 0:half_width]
    bottom_right = frame[half_height:height, half_width:width]
    
    return [top_left, top_right, bottom_left, bottom_right]

def rcut_img():
    frame_counter = 0  # 帧计数器
    start_time = time.time()  # 添加这行来初始化计时器
    
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            with condLock_empty:
                try:
                    condLock_empty.wait(timeout=0.5)
                except RuntimeError:
                    print("等待队列超时，继续检查...")
            continue
        
        # 获取各区域图像
        four_block = split_into_four_blocks(frame=frame)
        center_block = get_center_crop(frame=frame)
        
        # 处理各区域
        frame_tensor = preprocess_frame(frame=frame)
        center_block_tensor = preprocess_frame(frame=center_block)
        four_block_tensor_list = [preprocess_frame(frame=block) for block in four_block]
        
        # 区域预测并添加区域标识
        origin_pred = predict_tensor(frame_tensor)
        origin_pred["region"] = "full_frame"
        
        center_pred = predict_tensor(center_block_tensor)
        center_pred["region"] = "center_crop"
        
        block_preds = []
        for i, block_tensor in enumerate(four_block_tensor_list):
            pred = predict_tensor(block_tensor)
            pred["region"] = f"block_{i}"
            block_preds.append(pred)
            
        all_preds = [origin_pred, center_pred] + block_preds
        
        # 选择置信度最大的预测作为最终结果
        best_pred = max(all_preds, key=lambda x: x['confidence'])
        if(best_pred['confidence']<min_confidence):
            best_pred['class_name'] = "no exist!"
        print(f" {best_pred['class_name']} (ID: {best_pred['class_id']}, 置信度: {best_pred['confidence']:.4f})")
        
        result_frame = frame.copy()
        if(best_pred['confidence']>min_confidence):
            # 绘制文本标签
            cv2.putText(
                result_frame,
                f"{best_pred['class_name']} ({best_pred['confidence']:.4f}))",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # 计算边界框坐标
            height, width = frame.shape[:2]
            bbox_coords = None
            
            if best_pred["region"] == "full_frame":
                # 整个帧
                bbox_coords = (0, 0, width, height)
            elif best_pred["region"] == "center_crop":
                # 中心裁剪区域
                scale = 2/3  # 必须与get_center_crop函数使用的比例相同
                center_width = int(width * scale)
                center_height = int(height * scale)
                start_x = (width - center_width) // 2
                start_y = (height - center_height) // 2
                bbox_coords = (start_x, start_y, center_width, center_height)
            elif best_pred["region"].startswith("block_"):
                # 四个子区域之一
                block_index = int(best_pred["region"].split("_")[1])
                half_width = width // 2
                half_height = height // 2
                
                if block_index == 0:  # 左上
                    bbox_coords = (0, 0, half_width, half_height)
                elif block_index == 1:  # 右上
                    bbox_coords = (half_width, 0, half_width, half_height)
                elif block_index == 2:  # 左下
                    bbox_coords = (0, half_height, half_width, half_height)
                elif block_index == 3:  # 右下
                    bbox_coords = (half_width, half_height, half_width, half_height)
            
            # 绘制边界框
            if bbox_coords:
                x, y, w, h = bbox_coords
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # 创建固定大小的窗口而非全屏窗口
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)  # 使用AUTOSIZE而非NORMAL
        # 或者保持NORMAL但设置固定大小
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("result", 800, 600)  # 设置窗口初始大小
        
        cv2.imshow("result", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 帧计数和FPS监控
        frame_counter += 1
        if frame_counter % 30 == 0:  # 每30帧打印一次FPS
            elapsed_time = time.time() - start_time
            fps = 30 / elapsed_time
            print(f"处理速度: {fps:.2f} FPS")
            start_time = time.time()  # 重置计时器

def predict_tensor(processed_tensor):
    """
    使用预处理后的张量进行预测
    
    参数:
        processed_tensor: 已经预处理过的图像张量，通过preprocess_frame函数生成
        original_frame: 原始图像帧，如果需要可视化结果则提供
        
    返回:
        dict: 包含预测结果的字典，格式如下:
            {
                'class_name': 预测的类别名称,
                'class_id': 原始预测的类别ID
            }
    """
    model.eval()
    with paddle.no_grad():
        logits = model(processed_tensor)
        probabilities = paddle.nn.functional.softmax(logits, axis=1)
        pred_idx = paddle.argmax(logits, axis=1).numpy()[0]
        confidence = probabilities[0][pred_idx].numpy().item()
        result = {
            'class_id': int(pred_idx),
            'confidence': float(confidence),
        }
        if pred_idx in label_dict:
            idx_temp = label_dict[pred_idx]
            try:
                pred_class_name = result_name[int(idx_temp)]
                result['class_name'] = pred_class_name
            except:
                result['class_name'] = f"未知类别({idx_temp})"
                print(f"警告: 无法找到类别 ID {idx_temp} 的名称")
        else:
            result['class_name'] = f"未知类别({pred_idx})"
            print(f"警告: 未知的类别 ID: {pred_idx}")
        return result

if __name__ == "__main__":  
    rtsp_url = config.rtsp_url
    local_video_path = config.local_video_path
    frame_time = config.frame_time
    max_queue_size = config.max_queue_size
    
    # 从CSV文件加载标签
    result_name = {}  # 初始化空字典
    csv_path = "./dataset/labels.csv"
    result_name = read_labels_to_dict(csv_path)  # 保存返回值
    min_confidence =config.CONFIDENCE_THRESHOLD 
    print(f"从CSV加载了{len(result_name)}个标签")
    
    frame_queue = queue.Queue(maxsize=max_queue_size)
    condLock_full = threading.Condition()
    condLock_empty = threading.Condition()
    
    if hasattr(datasets['train'], 'label_dict') and datasets['train'].label_dict:
        label_dict = datasets['train'].label_dict
        print(f"从datasets加载了{len(label_dict)}个标签")
    
    try:
        model_path = "./work/mymodel/best_model.pdparams"
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            exit(1)
            
        num_classes = len(datasets['train'].label_dict)
        model = mobilenet_v3_small(num_classes=num_classes, pretrained=False)
        model_dict = paddle.load(model_path)
        model.load_dict(model_dict)
        model.eval()  
        print(f"模型加载成功，类别数: {num_classes}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        exit(1)
    
    cho = input("web视频流请输入1, 本地视频请输入2: ")
    
    capture_thread = None
    if cho == "1":
        user_rtsp = input(f"请输入RTSP URL (默认: {rtsp_url}): ")
        if user_rtsp:
            rtsp_url = user_rtsp
        capture_thread = threading.Thread(target=listen_web_carmer, daemon=True)
        capture_thread.start()
    else:
        user_path = input(f"请输入本地视频路径 (默认: {local_video_path}): ")
        if user_path:
            local_video_path = user_path
        
        if not os.path.exists(local_video_path):
            print(f"错误: 视频文件不存在: {local_video_path}")
            exit(1)
            
        capture_thread = threading.Thread(target=local_video, daemon=True)
        capture_thread.start()
    
    predict_thread = threading.Thread(target=rcut_img, daemon=True)
    predict_thread.start()
    
    try:
        while True:
            time.sleep(0.1)
            if not capture_thread.is_alive() and frame_queue.empty():
                print("捕获线程已结束且队列为空，准备退出...")
                break
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    
    print("程序退出")
