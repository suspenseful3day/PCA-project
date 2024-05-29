import cv2
import face_recognition

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 위치 찾기
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        # 얼굴 주변에 사각형 그리기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체와 창 닫기
cap.release()
cv2.destroyAllWindows()
