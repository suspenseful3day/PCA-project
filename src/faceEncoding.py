import cv2
import re
import os
import face_recognition
import pickle # 학습시킨 데이터를 pickle파일 형태로 저장

class faceEncoding():
    def __init__(self):
        self.dataset_paths = 'drive/MyDrive/linearAlgebra2_face_detection_datasets/'
        self.names = ['1']
        self.image_type = '.jpg' or '.JPG'
        self.encoding_file = 'encodings.pickle'

        # model의 두 가지가 있다. hog와 cnn
        # cnn은 높은 얼굴 인식 정확도를 보이지만 속도가 느리다는 단점(단, GPU환경은 빠르다)
        # hog는 낮은 얼굴 인식 정확도를 보이지만 속도가 빠르다는 장점(cnn-gpu와 속도 비슷)
        self.model_method = 'cnn-gpu' # GPU환경 사용, 일반 CPU환경은 cnn

        # feature dataset
        self.knownEncodings = []
        # label dataset
        self.knownNames = []

    def face_encoding_and_save(self):
        # 이미지 경로 반복
        dataset_file = os.listdir(self.dataset_paths)

        for team_file in dataset_file:
            path_name = os.path.join(self.dataset_paths, team_file)
            team_files = os.listdir(path_name)

            for name_file in team_files:
                path_name = os.path.join(self.dataset_paths, team_file, name_file)
                name_files = os.listdir(path_name)

                for img in name_files :
                    path_name = os.path.join(self.dataset_paths, team_file, name_file, img)
                    # 입력 이미지를 load하고 BGR에서 RGB로 변환
                    image = cv2.imread(path_name)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # 입력 이미지에서 얼굴에 해당하는 box의 (x, y) 좌표 감지
                    boxes = face_recognition.face_locations(rgb, model=self.model_method)

                    # 얼굴 임베딩 계산
                    encodings = face_recognition.face_encodings(rgb, boxes)

                    # 인코딩 반복
                    for encoding in encodings:
                        self.knownEncodings.append(encoding)
                        self.knownNames.append(name_file)
                      
        # pickle파일 형태로 데이터 저장
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(self.encoding_file, "wb")
        f.write(pickle.dumps(data))
        f.close()