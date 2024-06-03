import dlib
import cv2

class FaceCheck() :
    def __init__(self) :
        # CNN 기반 얼굴 탐지기 load
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
 

    def face_detecting(self, ret, frame) :
        if not ret :
            return
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # face detecting
        faces = self.cnn_face_detector(gray, 1)

        # bool list
        results = []

        # 얼굴 탐지 결과
        for face in faces:
            results.append(self.face_analyze(frame, face.rect))

        # 얼굴 주위 사각형 그리기
        for face, result in zip(faces, results):
            x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
            if result :
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else : # 등록되지 않은 경우
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imshow('Face Detection', frame)

        return result

    def face_analyze(self, frame, rect) :
        # 탐지 결과를 학습된 모델을 통해 확인해 등록 여부 판단

        U, S, VT = np.linalg.svd(rect, full_matrices=False)
        print(f"svd result: U={U}, S={S}, VT={VT}")

        # 등록되지 않은 경우 return False

        # 등록된 경우 return True
        return True
