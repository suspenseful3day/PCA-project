import face_recognition
import os
import pickle
import dlib

# dlib의 GPU 설정 확인
print("Using dlib version:", dlib.__version__)
print("GPU enabled:", dlib.DLIB_USE_CUDA)

# 학습 데이터 경로
data_path = r'C:\Users\JuSeong\Desktop\face_recognition\inface'
encoded_faces = {}

# 폴더 내 각 사람의 폴더를 탐색
for person_name in os.listdir(data_path):
    person_folder = os.path.join(data_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue

    face_encodings = []

    # 각 사람의 사진을 읽어 얼굴 인코딩
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            face_encodings.append(encodings[0])

    if face_encodings:
        encoded_faces[person_name] = face_encodings

# 인코딩된 얼굴 데이터를 파일로 저장
with open('encoded_faces.pkl', 'wb') as f:
    pickle.dump(encoded_faces, f)

print("Face encoding complete.")

