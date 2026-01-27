import cv2


class CameraCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def readFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def showFrame(self, windowName, frame):
        cv2.imshow(windowName, frame)

    def waitKey(self, delay=1):
        return cv2.waitKey(delay) & 0xFF

    def isOpened(self):
        return self.cap.isOpened()
