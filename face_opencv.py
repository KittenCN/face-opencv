#coding=utf8
import cv2
import time

print('Press Esc to exit')
faceCascade = cv2.CascadeClassifier('D:\opencv-3.2.0\data\haarcascades\haarcascade_frontalface_default.xml')
imgWindow = cv2.namedWindow('FaceDetect', cv2.WINDOW_NORMAL)

def detect_face():
    capInput = cv2.VideoCapture(0)
    # 避免处理时间过长造成画面卡顿
    nextCaptureTime = time.time()
    faces = []
    if not capInput.isOpened(): print('Capture failed because of camera')
    while 1:
        ret, img = capInput.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if nextCaptureTime < time.time():
            nextCaptureTime = time.time() + 0.1
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        if faces is not None:
            for x, y, w, h in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('FaceDetect', img)
        # 这是简单的读取键盘输入，27即Esc的acsii码
        if cv2.waitKey(1) & 0xFF == 27: break
    capInput.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_face()