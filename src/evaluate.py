import face_recognition
import pickle
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class FaceRecog:
    def __init__(self):
        self.encoding_file = 'data/preprocessing-encodings.pickle'
        self.unknown_name = 'Unknown'
        self.model_method = 'cnn'

        # Load the dataset
        self.data = pickle.loads(open(self.encoding_file, "rb").read())

        # Split the dataset into training and testing sets
        self.name_groups = self.split_dataset(self.data, test_size=0.3)
        # Evaluate the model
        y_true, y_pred = self.evaluate_model()

        self.plot_confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        labels = sorted(list(set(y_true) | set(y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        cm_display.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45, ha='right')
        plt.title('Confusion Matrix')
        plt.show()

    def split_dataset(self, data, test_size=0.3):
        encodings = data["encodings"]
        names = data["names"]

        # Parse filenames to extract the names
        name_to_encodings = defaultdict(list)
        for encoding, name in zip(encodings, names):
            # name_key = name.split('_')[1]  # Extract the name part
            name_key = name
            name_to_encodings[name_key].append((encoding, name))

        train_encodings = []
        train_names = []
        test_encodings = []
        test_names = []

        name_groups = {}

        # Split each group into training and testing sets
        for name_key, encodings_list in name_to_encodings.items():
            # if len(encodings_list) == 100:  # Only consider names with exactly 100 samples
            random.shuffle(encodings_list)
            print(len(encodings_list))
            split_index = int(len(encodings_list) * (1 - test_size))
            print(split_index)
            train_list = encodings_list[:split_index]
            test_list = encodings_list[split_index:]

            name_groups[name_key] = {
                'train': train_list,
                'test': test_list
            }

            for encoding, name in train_list:
                train_encodings.append(encoding)
                train_names.append(name)

            for encoding, name in test_list:
                test_encodings.append(encoding)
                test_names.append(name)

        return name_groups

    def evaluate_model(self):
        overall_correct_predictions = 0
        overall_total_predictions = 0
    
        y_true = []
        y_pred = []

        for name_key, groups in self.name_groups.items():
            train_list = groups['train']
            test_list = groups['test']

            # Prepare training data for this name
            train_encodings = [encoding for encoding, name in train_list]
            train_names = [name for encoding, name in train_list]

            # Prepare testing data for this name
            test_encodings = [encoding for encoding, name in test_list]
            test_names = [name for encoding, name in test_list]

            correct_predictions = 0
            total_predictions = len(test_encodings)



            for encoding, true_name in zip(test_encodings, test_names):
                matches = face_recognition.compare_faces(train_encodings, encoding, tolerance=0.4)  # Adjusted tolerance
                face_distances = face_recognition.face_distance(train_encodings, encoding)
                name = self.unknown_name

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = train_names[i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                if name == true_name:
                    correct_predictions += 1
                
                y_true.append(true_name)
                y_pred.append(name)

            accuracy = correct_predictions / total_predictions
            overall_correct_predictions += correct_predictions
            overall_total_predictions += total_predictions

            print(f"Accuracy for {name_key}: {accuracy:.2f} ({correct_predictions}/{total_predictions})")

        overall_accuracy = overall_correct_predictions / overall_total_predictions
        print(f"Overall model accuracy: {overall_accuracy:.2f} ({overall_correct_predictions}/{overall_total_predictions})")

        return y_true, y_pred

if __name__ == "__main__":
    fr = FaceRecog()
