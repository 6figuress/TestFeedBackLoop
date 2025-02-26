import numpy as np
import cv2
from colorBounds import *

# Load white color bounds
low_bound = ColorBound().min_white
up_bound = ColorBound().max_white

def get_imagefeatures(mask1_path, mask2_path):

    mask1_original = cv2.imread(mask1_path)
    mask2_original = cv2.imread(mask2_path)

    if mask1_original is None or mask2_original is None:
        return None, None

    hsv1 = cv2.cvtColor(mask1_original, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(mask2_original, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv1, low_bound, up_bound)
    mask2 = cv2.inRange(hsv2, low_bound, up_bound)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask1_draw = mask1_original.copy()
    mask2_draw = mask2_original.copy()
    print(len(contours1))
    print(len(contours2))
    largest_contour1 = max(contours1, key=cv2.contourArea) if contours1 else None
    largest_contour2 = max(contours2, key=cv2.contourArea) if contours2 else None

    if largest_contour1 is not None:
        cv2.drawContours(mask1_draw, [largest_contour1], -1, (0, 255, 0), 5)
    if largest_contour2 is not None:
        cv2.drawContours(mask2_draw, [largest_contour2], -1, (0, 255, 0), 5)

    cv2.imshow('Mask 1 Contours', mask1_draw)
    cv2.imshow('Mask 2 Contours', mask2_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return largest_contour1, largest_contour2

cnt1, cnt2 = get_imagefeatures('res/mask10.jpeg', 'res/mask1.jpeg')

if cnt1 is not None and cnt2 is not None:
    for c1,c2 in zip(cnt1,cnt2):
        if c1.all()==c2.all():
            print("Different contours")
            print(c1)
            print(c2)
    similarity_score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0)
    print(f"Image similarity: {similarity_score}")

# TODO: Implement multple contour matching
