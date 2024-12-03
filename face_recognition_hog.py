import cv2
import face_recognition
import pickle

# 인코딩된 얼굴 데이터 로드
with open('encoded_faces.pkl', 'rb') as f:
    encoded_faces = pickle.load(f)

# HOG 기반 얼굴 탐지기 사용
def detect_faces_hog(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_locations, face_encodings

# 카메라를 열어줌
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        break

    frame_count += 1
    
    # 매 5 프레임마다 얼굴 감지
    if frame_count % 5 == 0:
        face_locations, face_encodings = detect_faces_hog(frame)
        
        # 인식된 얼굴 처리
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = {}
            
            # 각 사람의 얼굴 인코딩과 비교
            for person_name, encodings in encoded_faces.items():
                results = face_recognition.compare_faces(encodings, face_encoding)
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                matches[person_name] = min(face_distances) if any(results) else float('inf')
            
            # 가장 유사한 사람의 이름 찾기
            best_match = min(matches, key=matches.get)
            
            # 얼굴 주위에 사각형 그리기 및 이름 표시
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, best_match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # 결과를 화면에 보여줌
    cv2.imshow('Face Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 윈도우를 해제
cap.release()
cv2.destroyAllWindows()
