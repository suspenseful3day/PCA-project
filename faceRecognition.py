import face_recognition
import cv2
import camera
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        #카메라 켜기
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []

        # 샘플사진을 올리고 이를 인식하는 방법을 학습
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # 변수 초기화 작업
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):
        # 비디오의 단일 프레임 캡처
        frame = self.camera.get_frame()

        # 얼굴 인식 처리를 위해 비디오 프레임을 1/4크기로 조정
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 이미지를 BGR 색상(OpenCV 사용)에서 RGB 색상(face_recognition 사용)으로 변환
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # 현재 비디오 프레임에서 모든 얼굴과 얼굴 인코딩 찾기
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # 알려진 얼굴(들)과 일치하는지 확인
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: 얼굴 간 거리를 얼마나 고려할 것인지. 낮을수록 엄격
                # (0.6 이 일반적으로 최상의 성능)
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # 결과 표시
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # 감지된 프레임을 1/4 크기로 조정했으므로 얼굴 위치를 다시 확장
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 얼굴 주위에 상자 그리기
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 얼굴 아래에 이름을 포함한 레이블 그리기
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # Motion JPEG을 사용 중이지만 OpenCV는 기본적으로 원시 이미지를 캡처하므로
        # 비디오 스트림을 올바르게 표시하기 위해 JPEG로 인코딩.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        # 프레임 표시
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # q키가 눌리면 루프 종료
        if key == ord("q"):
            break

    # 정리 작업
    cv2.destroyAllWindows()
    print('finish')