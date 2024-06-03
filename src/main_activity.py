import cv2

from face_check import FaceCheck
from collect import Collect

# 카메라를 통해 Frame 추출
class MainActivity() :
    # need to video setting
    def __init__(self) :
        # camera on
        self.cap = cv2.VideoCapture(0)
        self.face_check = FaceCheck()
        self.collect = Collect()
        self.is_quit = False

    def __del__(self):
        self.cap.release()
        cv2.destoryAllWindows()

    def start_capture(self) :
        while True :
            ref, frame = self.cap.read()
            self.face_check.face_detector(ref, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
