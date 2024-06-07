import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 카메라를 열어줌
cap = cv2.VideoCapture(0)

# 저장된 PCA 모델과 얼굴 인코딩 로드
with open("trained_pca_faces.pkl", "rb") as f:
    pca, pca_features, image_shape = pickle.load(f)

def preprocess_frame(frame, image_shape):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_shape).flatten()
    return resized

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        break
    
    # 얼굴 인코딩
    preprocessed_frame = preprocess_frame(frame, image_shape)
    pca_frame = pca.transform([preprocessed_frame])
    
    # 유사도 계산 (코사인 유사도)
    similarities = cosine_similarity(pca_frame, pca_features)
    best_match_index = np.argmax(similarities)
    similarity_percentage = similarities[0][best_match_index] * 100
    
    # 얼굴 주위에 사각형 그리기 및 유사도 퍼센트 표시
    cv2.putText(frame, f"Similarity: {similarity_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
    
    # 결과를 화면에 보여줌
    cv2.imshow('Face Recognition', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 윈도우를 해제
cap.release()
cv2.destroyAllWindows()
