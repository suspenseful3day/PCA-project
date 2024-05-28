import cv2
import timeit
import dlib

class FaceCheck() :
    def __init__(self) :
        # 얼굴 탐지를 위한 dlib의 얼굴 탐지기와 랜드마크 예측기 초기화
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 비디오 캡처 객체 생성
        self.cap = cv2.VideoCapture(0)

        '''
        # 가중치 파일 경로
        cascade_filename = 'haarcascade_frontalface_alt.xml'
        # 모델 불러오기
        self.cascade = cv2.CascadeClassifier(cascade_filename)

        # 영상 파일 
        cam = cv2.VideoCapture('sample.mp4')
        # 이미지 파일
        img = cv2.imread('sample.jpg')

        # 영상 탐지기
        self.video_detector(cam)
        '''
    def video_detector(self) :
        while True:
            # 비디오 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                break

    '''
    def video_detector(self, cam):
        while True:
            start_t = timeit.default_timer()
            # 알고리즘 시작 시점
            """ 알고리즘 연산 """
        
            # 캡처 이미지 불러오기
            ret,img = cam.read()
            # 영상 압축
            img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
            # 그레이 스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            # cascade 얼굴 탐지 알고리즘 
            results = self.cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,        # 이미지 피라미드 스케일 factor
                                           minNeighbors=5,          # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)          # 탐지 객체 최소 크기
                                           )
                                                                           
            for box in results:
                x, y, w, h = box
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
     
            """ 알고리즘 연산 """ 
            # 알고리즘 종료 시점
            terminate_t = timeit.default_timer()
            FPS = 'fps' + str(int(1./(terminate_t - start_t )))
            cv2.putText(img,FPS,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        
            # 영상 출력        
            cv2.imshow('facenet',img)
        
            if cv2.waitKey(1) > 0: 
                break
        '''
    def img_detector(self, img) :          
        # 영상 압축
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = self.cascade.detectMultiScale(gray,            # 입력 이미지
                                       scaleFactor= 1.5,# 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20,20)  # 탐지 객체 최소 크기
                                       )        
        
        for box in results:        
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
    
        # 사진 출력        
        cv2.imshow('facenet',img)  
        cv2.waitKey(10000)
