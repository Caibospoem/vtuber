import cv2

def capture_photo_from_camera():
    # 初始化摄像头
    camera = cv2.VideoCapture(0)  # 0代表默认摄像头

    if not camera.isOpened():
        print("无法打开摄像头")
        return

    # 从摄像头读取一帧图像
    ret, frame = camera.read()

    if ret:
        # 显示图像
        cv2.imshow('摄像头', frame)

        # 等待按键，如需自动拍照可省略
        cv2.waitKey(0)

        # 保存图像
        photo_path = '../res/std_face_1.jpg'
        cv2.imwrite(photo_path, frame)
        print(f"照片已保存至：{photo_path}")

    # 释放摄像头
    camera.release()
    cv2.destroyAllWindows()

# 调用函数进行拍照
capture_photo_from_camera()
