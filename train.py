import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2

from preprocess import load_images_from_folder
from features import extract_hog_features

# -----------------------------
# 1. Load dataset
# -----------------------------
cats_data, cats_labels = load_images_from_folder("data/train/cats", 0)
dogs_data, dogs_labels = load_images_from_folder("data/train/dogs", 1)

X = np.array(cats_data + dogs_data)
y = np.array(cats_labels + dogs_labels)

# -----------------------------
# 2. Feature extraction
# -----------------------------
X_features = extract_hog_features(X)

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. SVM pipeline
# -----------------------------
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1))
])

clf.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. Visualize predictions
# -----------------------------
for i in range(5):
    plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
    plt.title("Predicted: {} | Actual: {}".format(
        "Dog" if y_pred[i]==1 else "Cat",
        "Dog" if y_test[i]==1 else "Cat"
    ))
    plt.show()
