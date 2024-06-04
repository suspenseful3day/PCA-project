import cv2
import re
import os
import face_recognition
import pickle # 학습시킨 데이터를 pickle파일 형태로 저장

# 손흥민과 테디 브루시의 얼굴 사진 각 10장씩
dataset_paths = ['knowns/']
names = ['1']
number_images = 10
image_type = '.jpg' or '.JPG'
encoding_file = 'encodings.pickle'

# model의 두 가지가 있다. hog와 cnn
# cnn은 높은 얼굴 인식 정확도를 보이지만 속도가 느리다는 단점(단, GPU환경은 빠르다)
# hog는 낮은 얼굴 인식 정확도를 보이지만 속도가 빠르다는 장점(cnn-gpu와 속도 비슷)
model_method = 'cnn-gpu' # GPU환경 사용, 일반 CPU환경은 cnn

# feature dataset
knownEncodings = []
# label dataset
knownNames = []

# 이미지 경로 반복
dirname = 'knowns'
files = os.listdir(dirname)

pattern = r"^(inface_[A-Z]+)"

for (i, dataset_path) in enumerate(dataset_paths):
    # 사람 이름 추출
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext == '.jpg' or ext == '.JPG':
            pathname = os.path.join(dirname, filename)

            extracted_str = (str)[re.match(pattern, filename)]
            print(extracted_str)

            # 입력 이미지를 load하고 BGR에서 RGB로 변환
            image = cv2.imread(pathname)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 입력 이미지에서 얼굴에 해당하는 box의 (x, y) 좌표 감지
            boxes = face_recognition.face_locations(rgb, model=model_method)

            # 얼굴 임베딩 계산
            encodings = face_recognition.face_encodings(rgb, boxes)


            # 인코딩 반복
            for encoding in encodings:
                print(filename, extracted_str, encoding)
                knownEncodings.append(encoding)
                knownNames.append(extracted_str)
            
# pickle파일 형태로 데이터 저장
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()