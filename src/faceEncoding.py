import cv2
import os
import face_recognition
import pickle

class FaceEncoding:
    def __init__(self):
        self.dataset_paths = 'C:/Users/2jh09/Downloads/linearAlgebra2_face_detection_datasets/'
        self.names = ['1']
        self.image_type = '.jpg' or '.JPG'
        self.encoding_file = 'preprocessing-encodings.pickle'

        self.model_method = 'cnn-gpu' # GPU환경 사용, 일반 CPU환경은 cnn

        self.knownEncodings = []
        self.knownNames = []

    def face_encoding_and_save(self):
        dataset_file = os.listdir(self.dataset_paths)

        for team_file in dataset_file:
            print(team_file)
            path_name = os.path.join(self.dataset_paths, team_file)
            team_files = os.listdir(path_name)

            for name_file in team_files:
                print(name_file)
                path_name = os.path.join(self.dataset_paths, team_file, name_file)
                name_files = os.listdir(path_name)

                for img in name_files:
                    path_name = os.path.join(self.dataset_paths, team_file, name_file, img)

                    image = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
                    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
                    boxes = face_recognition.face_locations(rgb, model=self.model_method)
                    encodings = face_recognition.face_encodings(rgb, boxes)

                    for encoding in encodings:
                        self.knownEncodings.append(encoding)
                        self.knownNames.append(name_file)

        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(self.encoding_file, "wb")
        f.write(pickle.dumps(data))
        f.close()

face_encoder = FaceEncoding()
face_encoder.face_encoding_and_save()