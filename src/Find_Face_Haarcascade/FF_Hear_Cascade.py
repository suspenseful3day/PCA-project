import cv2

# Haar Cascade 파일 경로 설정
cascade_path = "haarcascade_frontalface_default.xml"

# OpenCV Haar Cascade 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(cascade_path)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 검출된 얼굴 영역 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체와 창 닫기
cap.release()
cv2.destroyAllWindows()
