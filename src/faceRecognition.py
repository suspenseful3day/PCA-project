import cv2
import face_recognition
import pickle
import camera
import numpy as np
from collections import defaultdict

class FaceRecog():
    def __init__(self):
        # self.encoding_file = 'data/encodings.pickle'
        self.encoding_file = 'data/preprocessing-encodings.pickle'
        self.unknown_name = 'Unknown'
        self.model_method = 'cnn-gpu'
        # 학습된 데이터 load
        self.data = pickle.loads(open(self.encoding_file, "rb").read())

        self.cam = camera.VideoCamera()

        self.name_cnt = defaultdict(int)

        while True:
            ret, frame = self.cam.get_frame()
            frame, name = self.FaceDetecting(frame)
            self.name_cnt[name] += 1
            # show the frame
            cv2.imshow("INFACE", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    def __del__(self):            
        # do a bit of cleanup
        # similarity_text = f"{name}: {similarity:.2f}" if similarity is not None else name
        # cv2.putText(image, similarity_text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
        white_image = np.ones((500, 800, 3), np.uint8) * 255
        
        if not self.name_cnt:
            name = "Unknown"
        else:
            name = max(self.name_cnt, key=self.name_cnt.get)
            _sum = sum(self.name_cnt.values())
            
            FRR = ((_sum - max(self.name_cnt.values())) / _sum) * 100
            
        text = f"Your name is {name}, FRR: {FRR:.2f}%"
        cv2.putText(white_image, text, (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.imshow("INFACE", white_image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break      

        cv2.destroyAllWindows()
        print('finish')

    def FaceDetecting(self, image):
        # start_time = time.time()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 입력 이미지에서 각 얼굴에 해당하는 box 좌표를 감지하고 얼굴 임베딩 계산
        boxes = face_recognition.face_locations(rgb, model=self.model_method)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # 감지된 각 얼굴의 이름 목록 초기화
        names = []

        # 유사도 추가
        similarities = []

        # 얼굴 임베딩 반복
        for encoding in encodings:
            # 입력 이미지의 각 얼굴과 학습된 데이터 매치
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=0.3)
            face_distances = face_recognition.face_distance(self.data["encodings"], encoding)
            name = self.unknown_name
            similarity = None

            # 데이터가 매치된 경우
            if True in matches:
                # 일치하는 모든 얼굴의 인덱스를 찾고, 얼굴 일치 횟수 계산을 위한 초기화
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # 유사도 계산을 위해 최소 거리 선택
                best_match_index = face_distances.argmin()

                # 일치하는 인덱스를 반복하고, 인식된 각 얼굴의 수 유지
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # 가장 많은 표를 얻은 label 선택
                # print(counts)
                name = max(counts, key=counts.get)
                similarity = 1 - face_distances[best_match_index]  # 유사도 계산 (1 - 거리)

            # 이름 목록 업데이트
            names.append(name)
            # 유사도 목록 업데이트
            similarities.append(similarity)

        # 인식된 얼굴 반복
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # 이미지에 인식된 얼굴의 box를 그림
            y = top - 15 if top - 15 > 15 else top + 15
            # 학습된 얼굴(인식한 얼굴)의 경우 녹색 선
            color = (0, 255, 0)
            line = 2
            # 학습되지 않은 얼굴(인식하지 못한 얼굴)의 경우 빨간색 선
            if(name == self.unknown_name):
                color = (0, 0, 255)
                line = 1
                name = 'Unknown'
            # 사각형 그리기
            cv2.rectangle(image, (left, top), (right, bottom), color, line)
            y = top - 15 if top - 15 > 15 else top + 15
            # 텍스트 추가
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color, line)
            # 텍스트 추가
            similarity_text = f"{name}: {similarity:.2f}" if similarity is not None else name
            cv2.putText(image, similarity_text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
        
        # end_time = time.time()
        # 소요시간 체크
        # process_time = end_time - start_time

        if not names:
            names.append('Unknown')

        return image, names[0]