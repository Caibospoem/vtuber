import time
import logging
import threading
import multiprocessing

import cv2
import dlib
import numpy as np
from rimo_utils import 计时

def 多边形面积(a):
    a = np.array(a)
    x = a[:, 0]
    y = a[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')

def get_key_points(img, face_position):
    landmark_shape = predictor(img, face_position)
    key_points = []
    for i in range(68):
        pos = landmark_shape.part(i)
        key_points.append(np.array([pos.x, pos.y], dtype=np.float32))
    return np.array(key_points)


def 计算旋转量(key_points):
    def 中心(索引数组):
        return sum([key_points[i] for i in 索引数组]) / len(索引数组)
    左眉 = [18, 19, 20, 21]
    右眉 = [22, 23, 24, 25]
    下巴 = [6, 7, 8, 9, 10]
    鼻子 = [29, 30]
    眉中心, 下巴中心, 鼻子中心 = 中心(左眉 + 右眉), 中心(下巴), 中心(鼻子)
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    中线长 = np.linalg.norm(中线)
    横旋转量 = np.cross(中线, 斜边) / 中线长**2
    竖旋转量 = 中线 @ 斜边 / 中线长**2
    Z旋转量 = np.cross(中线, [0, 1]) / 中线长
    return np.array([横旋转量, 竖旋转量, Z旋转量])


def 计算嘴大小(key_points):
    边缘 = key_points[0:17]
    嘴边缘 = key_points[48:60]
    嘴大小 = 多边形面积(嘴边缘) / 多边形面积(边缘)
    return np.array([嘴大小])


def 计算相对位置(img, face_position):
    x = (face_position.top() + face_position.bottom())/2/img.shape[0]
    y = (face_position.left() + face_position.right())/2/img.shape[1]
    y = 1 - y
    相对位置 = np.array([x, y])
    return 相对位置


def 计算脸大小(key_points):
    边缘 = key_points[0:17]
    t = 多边形面积(边缘)**0.5
    return np.array([t])


def 计算眼睛大小(key_points):
    边缘 = key_points[0:17]
    左 = 多边形面积(key_points[36:42]) / 多边形面积(边缘)
    右 = 多边形面积(key_points[42:48]) / 多边形面积(边缘)
    return np.array([左, 右])


def 计算眉毛高度(key_points):
    边缘 = key_points[0:17]
    左 = 多边形面积([*key_points[18:22]]+[key_points[38], key_points[37]]) / 多边形面积(边缘)
    右 = 多边形面积([*key_points[22:26]]+[key_points[44], key_points[43]]) / 多边形面积(边缘)
    return np.array([左, 右])


detector = dlib.get_frontal_face_detector()

def positioning_face(img):
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))

def getdata(img):
    face_position = positioning_face(img)
    if not face_position:
        return None
    相对位置 = 计算相对位置(img, face_position)
    key_points = get_key_points(img, face_position)
    旋转量组 = 计算旋转量(key_points)
    脸大小 = 计算脸大小(key_points)
    眼睛大小 = 计算眼睛大小(key_points)
    嘴大小 = 计算嘴大小(key_points)
    眉毛高度 = 计算眉毛高度(key_points)
    
    img //= 2
    img[face_position.top():face_position.bottom(), face_position.left():face_position.right()] *= 2
    for i, (px, py) in enumerate(key_points):
        cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    
    return np.concatenate([旋转量组, 相对位置, 嘴大小, 脸大小, 眼睛大小, 眉毛高度])

# 初始化脸部原始特征
orig_image_features = getdata(cv2.imread('../res/std_face_1.jpg'))
image_features = orig_image_features - orig_image_features


def feature_capture(pipe):
    global orig_image_features
    global image_features
    cap = cv2.VideoCapture(0)
    logging.warning('开始捕捉了！')
    while True:
        with 计时.帧率计('提特征'):
            ret, img = cap.read()
            new_image_features = getdata(img)
            cv2.imshow('', img[:, ::-1])
            cv2.waitKey(1)
            if new_image_features is not None:
                image_features = new_image_features - orig_image_features
            pipe.send(image_features)
            print('特征点为', image_features)


def get_img_features():
    global image_features
    return image_features


def trans():
    global image_features
    logging.warning('转移线程启动了！')
    while True:
        image_features = pipe[1].recv() # 更新 image_features


pipe = multiprocessing.Pipe() # 创建一个多进程管道


def start():
    t = threading.Thread(target=trans) # 新建一个线程对象t，t线程启动后要执行的函数为 转移
    t.setDaemon(True) # 设置为守护线程，随主程序的关闭而关闭
    t.start() # 启动线程，执行转移
    logging.warning('捕捉进程启动中……')
    p = multiprocessing.Process(target=feature_capture, args=(pipe[0],))# 新建一个线程对象t，t线程启动后要执行的函数为 捕捉循环
    p.daemon = True # 设置为守护线程，随主程序的关闭而关闭
    p.start()


if __name__ == '__main__':
    start()
    np.set_printoptions(precision=3, suppress=True)
    while True:
        time.sleep(0.1)
        print(get_img_features())
