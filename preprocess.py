import os
import cv2

def load_images_from_folder(folder, label, img_size=(64,64)):
    data, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, img_size)
            data.append(img)
            labels.append(label)
    return data, labels
