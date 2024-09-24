import os
from random import shuffle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    images = []
    labels = []
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 调整大小
            height, width = img.shape[:2]
            target_size = 128
            
            scale = target_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            
            img_resized = cv2.resize(img, new_size)

            delta_w = target_size - img_resized.shape[1]
            delta_h = target_size - img_resized.shape[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            images.append(img_padded)
            labels.append(class_index)

    return np.array(images).reshape(-1, 128, 128, 1), np.array(labels), class_names

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_dir = os.getcwd()  
    images, labels, class_names = load_data(data_dir)  # 获取数据和类名
    
    # 打乱数据
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # 数据生成器设置
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  
    
    # 训练集生成器
    train_generator = datagen.flow(images, labels, batch_size=32, shuffle=True, subset='training')
    
    # 验证集生成器
    validation_generator = datagen.flow(images, labels, batch_size=32, shuffle=False, subset='validation')

    # 训练模型
    model = create_cnn_model(input_shape=(128, 128, 1), num_classes=len(class_names))
    model.fit(train_generator, epochs=10, validation_data=validation_generator) 

    # 评估模型
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'验证集损失: {test_loss:.4f}, 验证集准确率: {test_acc:.4f}')

    # 导出模型
    model.save('cnn_model.h5')
    

if __name__ == "__main__":
    main()
