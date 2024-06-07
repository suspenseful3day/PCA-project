import os
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA

# 학습 데이터 경로 설정
training_images_path = r"C:\Users\JuSeong\Desktop\pca_face_recognition\inface_PJS"  # 이미지들이 저장된 경로

# 이미지 로드 및 전처리 함수
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

# 이미지 데이터를 로드
images = load_images_from_folder(training_images_path)

# 각 이미지 크기를 동일하게 조정 (예: 100x100)
image_shape = (100, 100)
images_resized = [cv2.resize(img, image_shape).flatten() for img in images]

# PCA 모델 학습
n_components = 50  # 주성분 개수
pca = PCA(n_components=n_components, whiten=True).fit(images_resized)

# 주성분으로 변환된 얼굴 데이터
pca_features = pca.transform(images_resized)

# 얼굴 인코딩과 이름을 파일로 저장
with open("trained_pca_faces.pkl", "wb") as f:
    pickle.dump((pca, pca_features, image_shape), f)

print("훈련 완료: PCA 기반 얼굴 인코딩이 저장되었습니다.")
