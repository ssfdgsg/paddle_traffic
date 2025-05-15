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

def 


def predict():
    window_created = False
    model.eval()
    with paddle.no_grad():
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                processed_frame = preprocess_frame(frame)
                logits = model(processed_frame)
                probabilities = paddle.nn.functional.softmax(logits, axis=1)
                pred_idx = paddle.argmax(logits, axis=1).numpy()[0]
                confidence = probabilities[0][pred_idx].numpy().item()
                if pred_idx in label_dict:
                    idx_temp = 0
                    idx_temp = label_dict[pred_idx]
                    try:
                        pred_class_name = result_name[int(idx_temp)]
                    except:
                        print(idx_temp)
                    print(f"检测到: {pred_class_name} (ID: {pred_idx}, 置信度: {confidence:.4f})")
                    result_frame = frame.copy()
                    cv2.putText(
                        result_frame,
                        f"{pred_class_name} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", result_frame)
                    window_created = True
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f"unknown: {pred_idx}")
                    
            except queue.Empty:
                with condLock_empty:
                    try:
                        condLock_empty.wait(timeout=0.5)
                    except RuntimeError:
                        print("等待队列超时，继续检查...")
                continue
            except Exception as e:
                print(f"预测线程出错: {e}")
                import traceback
                traceback.print_exc()
                break
    
    if window_created:
        try:
            cv2.destroyWindow("预测结果")
        except:
            pass
    print("预测线程结束")

if __name__ == "__main__":  
    rtsp_url = config.rtsp_url
    local_video_path = config.local_video_path
    frame_time = config.frame_time
    max_queue_size = config.max_queue_size
    
    # 从CSV文件加载标签
    result_name = {}  # 初始化空字典
    csv_path = "./dataset/labels.csv"
    result_name = read_labels_to_dict(csv_path)  # 保存返回值
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
    
    predict_thread = threading.Thread(target=predict, daemon=True)
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