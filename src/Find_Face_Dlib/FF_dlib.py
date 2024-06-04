import cv2
import dlib

# 얼굴 탐지를 위한 dlib의 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 히스토그램 평활화 적용
    gray = cv2.equalizeHist(gray)

    # 얼굴 탐지 (업샘플링 횟수 증가)
    faces = detector(gray, 1)  # upsample_num_times=1

    for face in faces:
        # 얼굴 영역 그리기
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 얼굴 랜드마크 예측
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # 결과 프레임 보여주기
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체와 창 닫기
cap.release()
cv2.destroyAllWindows()
