import os
import cv2
import numpy as np
from skimage import feature
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

def preprocess_image(image_path):
    """Reads and preprocesses an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    img = cv2.resize(img, (64, 64))
    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    equalized_img = cv2.equalizeHist(normalized_img)
    return equalized_img

def get_data(directory):
    """Loads images and labels from a given directory."""
    labels = []
    images = []
    label = 1 if 'female' in directory else 0  # 1 for female, 0 for male

    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        images.append(preprocess_image(img_path))
        labels.append(label)

    return images, labels

def combine_data(base_path):
    """Combines male and female image data from the base path."""
    male_path = os.path.join(base_path, 'male')
    female_path = os.path.join(base_path, 'female')

    male_data, male_labels = get_data(male_path)
    female_data, female_labels = get_data(female_path)

    return male_data + female_data, male_labels + female_labels

def extract_lbp_features(images, radius, num_points):
    """Extracts LBP features from a list of images."""
    features = []
    for image in images:
        lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
    return features

def train_svm(data, labels, kernel='linear'):
    """Trains an SVM classifier with the given kernel."""
    clf = svm.SVC(kernel=kernel)
    if kernel == 'linear':
        clf.fit(data, labels)
    else:
        param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001, 0.00001, 10]}
        clf = GridSearchCV(clf, param_grid)
        clf.fit(data, labels)

    return clf

def test_svm(clf, data, labels):
    """Tests an SVM classifier and returns the accuracy and predictions."""
    predictions = clf.predict(data)
    accuracy = accuracy_score(labels, predictions) * 100
    return predictions, accuracy

def write_results_to_file(predictions, labels, accuracy, file_name="results.txt"):
    """Writes the test results to a text file."""
    confusion = confusion_matrix(labels, predictions).ravel()
    with open(file_name, "w") as file:
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write(f"Confusion Matrix: {confusion}\n")

def process_data(train_path, val_path, test_path):
    """Main function to process the data and run the SVM classifier."""
    train_data, train_labels = combine_data(train_path)
    val_data, val_labels = combine_data(val_path)
    test_data, test_labels = combine_data(test_path)

    # Extract LBP features
    radius_points = (1, 8)  # Example radius and points
    train_features = extract_lbp_features(train_data, *radius_points)
    val_features = extract_lbp_features(val_data, *radius_points)
    test_features = extract_lbp_features(test_data, *radius_points)

    # Train and test SVM
    clf = train_svm(train_features + val_features, train_labels + val_labels)
    predictions, accuracy = test_svm(clf, test_features, test_labels)

    # Write results
    write_results_to_file(predictions, test_labels, accuracy)

if __name__ == '__main__':
    print('Starting the classification process...')
    args = sys.argv[1:]
    args.sort()
    process_data(*args)
    print('Classification complete. Check the results in the output file.')
