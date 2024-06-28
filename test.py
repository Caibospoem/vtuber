import numpy as np
import time
import multiprocessing
import dlib
import cv2
from get_feature import start, get_img_features, positioning_face, feature_capture, get_key_points

if __name__ == '__main__':
    img = cv2.imread('../res/std_face_1.jpg')
    predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')
    pos = positioning_face(img)
    print(get_key_points(img, pos))
    print(get_key_points(img, pos).shape)