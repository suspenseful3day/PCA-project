# 필요한 패키지 설치
!pip install opencv-python
!pip install face_recognition

# 패키지 임포트
import cv2
import re
import os
import face_recognition
import pickle
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

class faceEncoding():
    def __init__(self):
        self.dataset_paths = ['/content/drive/MyDrive/knowns']
        self.encoding_file = '/content/drive/MyDrive/encodings.pickle'  # 경로 수정

        # model의 두 가지가 있다. hog와 cnn
        # cnn은 높은 얼굴 인식 정확도를 보이지만 속도가 느리다는 단점(단, GPU환경은 빠르다)
        # hog는 낮은 얼굴 인식 정확도를 보이지만 속도가 빠르다는 장점(cnn과 속도 비슷)
        self.model_method = 'cnn'  # 'cnn-gpu' 대신 'cnn' 사용

        # feature dataset
        self.knownEncodings = []
        # label dataset
        self.knownNames = []

    def face_encoding_and_save(self):
        # 각 사람의 폴더를 반복
        for dataset_path in self.dataset_paths:
            for person_folder in os.listdir(dataset_path):
                person_path = os.path.join(dataset_path, person_folder)
                if os.path.isdir(person_path):
                    print(f"Processing folder: {person_folder}")
                    
                    # 사람 이름 추출
                    extracted_str = person_folder
                    print(f"Extracted string: {extracted_str}")

                    # 폴더 내의 각 이미지 파일을 반복
                    for filename in os.listdir(person_path):
                        name, ext = os.path.splitext(filename)
                        if ext.lower() == '.jpg':
                            pathname = os.path.join(person_path, filename)
                            print(f"Full path: {pathname}")

                            # 입력 이미지를 load하고 BGR에서 RGB로 변환
                            image = cv2.imread(pathname)
                            if image is None:
                                print(f"Error loading image: {pathname}")
                                continue
                            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            print(f"Image loaded and converted to RGB")

                            # 입력 이미지에서 얼굴에 해당하는 box의 (x, y) 좌표 감지
                            boxes = face_recognition.face_locations(rgb, model=self.model_method)
                            print(f"Face locations: {boxes}")

                            # 얼굴 임베딩 계산
                            encodings = face_recognition.face_encodings(rgb, boxes)
                            print(f"Encodings: {encodings}")

                            # 인코딩 반복
                            for encoding in encodings:
                                print(f"Filename: {filename}, Extracted String: {extracted_str}, Encoding: {encoding}")
                                self.knownEncodings.append(encoding)
                                self.knownNames.append(extracted_str)

        # pickle파일 형태로 데이터 저장
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        with open(self.encoding_file, "wb") as f:
            f.write(pickle.dumps(data))
        print("Encodings saved to file")

# faceEncoding 클래스 인스턴스 생성 및 메서드 호출
if __name__ == "__main__":
    face_encoder = faceEncoding()
    face_encoder.face_encoding_and_save()
