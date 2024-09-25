import os
from warnings import filters
import numpy as np
import cv2
import hashlib
from skimage import feature
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from skimage import color
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import joblib

def hash_image(image):
    """计算图像的哈希值"""
    img_flattened = image.flatten()  
    return hashlib.md5(img_flattened).hexdigest()  

def create_densenet_model(img):
    """提取图像的GIST特征"""
    
    # 定义滤波器的参数
    orientations = 8  # 方向数
    scales = [1, 2, 4, 8]  # 尺度
    gist_features = []

    for scale in scales:
        blurred = filters.gaussian(img, sigma=scale)
        
        # 计算梯度
        gradient_x = np.gradient(blurred, axis=0)
        gradient_y = np.gradient(blurred, axis=1)
        
        # 计算梯度的方向和幅度
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x) + np.pi 
        
        # 生成方向直方图
        hist, _ = np.histogram(direction, bins=orientations, range=(0, 2 * np.pi), weights=magnitude)
        
        # 归一化特征
        gist_features.append(hist / np.sum(hist + 1e-6))

    return np.concatenate(gist_features)


def load_data():
    images, labels = [], []  
    class_names = [d for d in os.listdir() if os.path.isdir(d)] 
    fixed_size = (256,256)

    print("开始加载数据...")

    for class_name in class_names:
        print(f"开始加载 {class_name}")
        class_dir = os.path.join(os.getcwd(), class_name)  
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)  
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  

            if img is None:
                print(f"警告: 无法加载图像 {img_path}")
                continue  

            img_resized = cv2.resize(img, fixed_size)  
            gist_features = extract_gist_features(img_resized)  
            images.append(gist_features) 
            labels.append(class_names.index(class_name))  
        print(f"加载完成{class_name}")

    print("数据加载完成")
    return np.array(images), np.array(labels), class_names  

def main():
    images, labels, class_names = load_data()  
    print(f"加载的图像数量: {len(images)}")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  
    accuracies = []  

    for fold, (train_index, val_index) in enumerate(kf.split(images), 1):
        print(f"\n第 {fold} 折交叉验证:")
        X_train, X_val = images[train_index], images[val_index]  
        y_train, y_val = labels[train_index], labels[val_index]

        knn = KNeighborsClassifier(n_neighbors=3)  
        knn.fit(X_train, y_train)  # 训练模型
        print("模型训练完成!")

        y_pred = knn.predict(X_val)  
        accuracy = accuracy_score(y_val, y_pred)  
        accuracies.append(accuracy)  

        print(f"验证集准确率: {accuracy:.3f}")

    print(f'\n平均准确率: {np.mean(accuracies):.3f}')  

    joblib.dump(knn, 'knn_model.pkl')
    print("模型已导出到 knn_model.pkl")

if __name__ == "__main__":
    main()
