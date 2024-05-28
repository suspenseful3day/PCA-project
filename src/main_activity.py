import cv2
import timeit

from face_check import FaceCheck

class MainActivity() :
    # need to video setting
    def __init__(self) :
        self.face_check = FaceCheck()