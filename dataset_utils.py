import os
from PIL import Image
import paddle
from paddle.vision import transforms


# 导入 PaddleX 中的 transforms 模块，这个模块包含了多种图像预处理的方法
from paddle.vision import transforms
train_transforms = transforms.Compose([
    transforms.Resize(256),  # 首先调整到固定大小
    transforms.RandomCrop(224),  # 然后随机裁剪
    transforms.RandomHorizontalFlip(prob=0.5),  # 水平翻转
    transforms.RandomRotation(15),  # 随机旋转，增强模型鲁棒性
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 色彩扰动
    transforms.RandomErasing(prob=0.2, scale=(0.02, 0.2)),  # 随机擦除
    transforms.Transpose(),  # CHW格式转换
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),  # 首先调整到固定大小
    transforms.CenterCrop(224),  # 然后中心裁剪
    transforms.Transpose(),  # CHW格式转换
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 定义用于ImageNet格式数据的自定义数据集
class ImageNetDataset(paddle.io.Dataset):
    def __init__(self, data_dir, file_list, label_list, transforms=None, shuffle=False):
        super(ImageNetDataset, self).__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        
        # 读取文件列表
        self.file_paths = []
        self.labels = []
        with open(file_list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保行格式正确
                    self.file_paths.append(parts[0])
                    self.labels.append(int(parts[1]))
                    
        # 读取标签映射
        self.label_dict = {}
        try:
            with open(label_list, 'r') as f:
                for i, line in enumerate(f):
                    self.label_dict[i] = line.strip()
        except Exception as e:
            print(f"警告: 无法读取标签文件: {e}")
            # 如果无法读取标签文件，创建默认标签
            for i in set(self.labels):
                self.label_dict[i] = str(i)
                
        # 验证标签范围
        max_label = max(self.labels) if self.labels else 0
        if max_label >= len(self.label_dict):
            print(f"警告: 最大标签值 {max_label} 超出标签字典大小 {len(self.label_dict)}")
            # 扩展标签字典以包含所有标签
            for i in range(len(self.label_dict), max_label + 1):
                self.label_dict[i] = str(i)
                
        if shuffle:
            import random
            combined = list(zip(self.file_paths, self.labels))
            random.shuffle(combined)
            self.file_paths, self.labels = zip(*combined)
            self.file_paths = list(self.file_paths)  # 转回列表
            self.labels = list(self.labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.file_paths[index])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像出错 {img_path}: {e}")
            # 返回一个黑色图像作为替代
            img = Image.new('RGB', (64, 64), color='black')
            
        if self.transforms:
            img = self.transforms(img)
            
        return img, self.labels[index]
    
    def __len__(self):
        return len(self.file_paths)

# 数据目录和文件路径
data_dir = './dataset/traffic_Data/DATA'
train_list = os.path.join(data_dir, 'train_list.txt')
val_list = os.path.join(data_dir, 'val_list.txt')
label_list = os.path.join(data_dir, 'labels.txt')
test_list = os.path.join(data_dir,'test_list.txt')

# 定义训练数据集
train_dataset = ImageNetDataset(
    data_dir=data_dir,
    file_list=train_list,
    label_list=label_list,
    transforms=train_transforms,
    shuffle=True)

# 定义评估数据集
eval_dataset = ImageNetDataset(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=eval_transforms)

test_dataset = ImageNetDataset(
    data_dir=data_dir,
    file_list=test_list,
    label_list=label_list,
    transforms=eval_transforms)

# 将数据集整合到字典中便于访问
datasets = {
    'train': train_dataset,
    'eval': eval_dataset,
    'test':test_dataset
}

# 创建数据加载器
batch_size = 32
loaders = {
    k: paddle.io.DataLoader(
        v, 
        batch_size=batch_size, 
        shuffle=(k=='train')
    ) for k, v in datasets.items()
}

# 打印数据集信息
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(eval_dataset)}")
print(f"测试集数量:{len(test_dataset)}")
print(f"类别数: {len(train_dataset.label_dict)}")