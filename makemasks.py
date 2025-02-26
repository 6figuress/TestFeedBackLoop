from colorBounds import *
import cv2

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

low_bound = ColorBound().min_red
up_bound = ColorBound().max_red

def get_images(img_path, mask_path,contours_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_bound, up_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2000:
            cv2.drawContours(img, c, -1, (0, 255, 0), 10)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(contours_path, img)
    return img,mask



if __name__ == "__main__":
    i=0
    for filename in os.listdir('img'):
        img_path = os.path.join('img', filename)
        mask_path = f'res/mask{i}.jpeg'
        contours_path = f'res/contours{i}.jpeg'
        i+=1
        img, mask = get_images(img_path, mask_path, contours_path)
        




