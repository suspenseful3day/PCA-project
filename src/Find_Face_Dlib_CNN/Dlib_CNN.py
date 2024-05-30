import cv2
import dlib

# CNN 기반 얼굴 탐지기 로드
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# 카메라를 열어줌
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        break
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    faces = cnn_face_detector(gray, 1)
    
    # 얼굴 주위에 사각형 그리기
    for face in faces:
        x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # 결과를 화면에 보여줌
    cv2.imshow('Face Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 윈도우를 해제
cap.release()
cv2.destroyAllWindows()
