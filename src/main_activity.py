import cv2

from face_check import FaceCheck
from collect import Collect

# 카메라를 통해 Frame 추출
class MainActivity() :
    # need to video setting
    def __init__(self) :
        # camera on
        self.cap = cv2.VideoCapture(1)
        self.face_check = FaceCheck()
        self.collect = Collect()
        
        self.start_capture()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def start_capture(self) :
        print(f"start capture")
        while True :
            ref, frame = self.cap.read()
            self.face_check.face_detecting(ref, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
