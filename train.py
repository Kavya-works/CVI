import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
import time

def load_images(folder_path, label, img_size=(64, 64)):
    images = []
    labels = []
    
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                img_flattened = img_resized.flatten()
                images.append(img_flattened)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
train_cat_folder = os.path.join(base_path, "train", "Cat")
train_dog_folder = os.path.join(base_path, "train", "Dog")
test_cat_folder = os.path.join(base_path, "test", "Cat")
test_dog_folder = os.path.join(base_path, "test", "Dog")

print("Loading images...")
train_cat_images, train_cat_labels = load_images(train_cat_folder, 0)
train_dog_images, train_dog_labels = load_images(train_dog_folder, 1)
test_cat_images, test_cat_labels = load_images(test_cat_folder, 0)
test_dog_images, test_dog_labels = load_images(test_dog_folder, 1)

X_train = np.vstack([train_cat_images, train_dog_images])
y_train = np.hstack([train_cat_labels, train_dog_labels])
X_test = np.vstack([test_cat_images, test_dog_images])
y_test = np.hstack([test_cat_labels, test_dog_labels])

print(f"Training: {X_train.shape[0]} images")
print(f"Test: {X_test.shape[0]} images")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# 1. Logistic Regression
print("\n=== Logistic Regression ===")
best_lr_score = 0
best_lr_model = None
best_lr_params = None

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    start_time = time.time()
    lr = LogisticRegression(C=C, max_iter=2000, random_state=42, solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    score = lr.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"C={C}: Accuracy = {score:.4f} (Time: {train_time:.2f}s)")
    
    if score > best_lr_score:
        best_lr_score = score
        best_lr_model = lr
        best_lr_params = {'C': C}

results['Logistic Regression'] = {
    'accuracy': best_lr_score,
    'model': best_lr_model,
    'params': best_lr_params
}
print(f"Best Logistic Regression Accuracy: {best_lr_score:.4f}")

# 2. Neural Network (MLP)
print("\n=== Neural Network (MLP) ===")
best_nn_score = 0
best_nn_model = None
best_nn_params = None

architectures = [
    (50,),      # Single layer, 50 neurons
    (100,),     # Single layer, 100 neurons
    (128,),     # Single layer, 128 neurons
    (50, 50),   # Two layers, 50 neurons each
    (100, 50),  # Two layers, 100 and 50 neurons
    (128, 64),  # Two layers, 128 and 64 neurons
    (256, 128), # Two layers, 256 and 128 neurons
]

for hidden_layers in architectures:
    start_time = time.time()
    nn = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, 
                       random_state=42, early_stopping=True, validation_fraction=0.1)
    nn.fit(X_train_scaled, y_train)
    score = nn.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"Layers {hidden_layers}: Accuracy = {score:.4f} (Time: {train_time:.2f}s)")
    
    if score > best_nn_score:
        best_nn_score = score
        best_nn_model = nn
        best_nn_params = {'hidden_layers': hidden_layers}

results['Neural Network'] = {
    'accuracy': best_nn_score,
    'model': best_nn_model,
    'params': best_nn_params
}
print(f"Best Neural Network Accuracy: {best_nn_score:.4f}")

# 3. Support Vector Machine (SVM)
print("\n=== Support Vector Machine (SVM) ===")
best_svm_score = 0
best_svm_model = None
best_svm_params = None

# Use smaller sample for SVM due to computational cost
sample_size = min(1000, len(X_train_scaled))
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_svm = X_train_scaled[indices]
y_train_svm = y_train[indices]

for C in [0.1, 1.0, 10.0]:
    for kernel in ['rbf', 'linear']:
        start_time = time.time()
        svm = SVC(C=C, kernel=kernel, random_state=42, probability=True)
        svm.fit(X_train_svm, y_train_svm)
        score = svm.score(X_test_scaled, y_test)
        train_time = time.time() - start_time
        print(f"C={C}, Kernel={kernel}: Accuracy = {score:.4f} (Time: {train_time:.2f}s)")
        
        if score > best_svm_score:
            best_svm_score = score
            best_svm_model = svm
            best_svm_params = {'C': C, 'kernel': kernel}

results['SVM'] = {
    'accuracy': best_svm_score,
    'model': best_svm_model,
    'params': best_svm_params
}
print(f"Best SVM Accuracy: {best_svm_score:.4f}")

# 4. Random Forest
print("\n=== Random Forest ===")
best_rf_score = 0
best_rf_model = None
best_rf_params = None

for n_estimators in [50, 100, 200]:
    for max_depth in [10, 20, None]:
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        score = rf.score(X_test_scaled, y_test)
        train_time = time.time() - start_time
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: Accuracy = {score:.4f} (Time: {train_time:.2f}s)")
        
        if score > best_rf_score:
            best_rf_score = score
            best_rf_model = rf
            best_rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

results['Random Forest'] = {
    'accuracy': best_rf_score,
    'model': best_rf_model,
    'params': best_rf_params
}
print(f"Best Random Forest Accuracy: {best_rf_score:.4f}")

# 5. K-Nearest Neighbors (KNN)
print("\n=== K-Nearest Neighbors (KNN) ===")
best_knn_score = 0
best_knn_model = None
best_knn_params = None

for n_neighbors in [3, 5, 7, 9, 11]:
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"n_neighbors={n_neighbors}: Accuracy = {score:.4f} (Time: {train_time:.2f}s)")
    
    if score > best_knn_score:
        best_knn_score = score
        best_knn_model = knn
        best_knn_params = {'n_neighbors': n_neighbors}

results['KNN'] = {
    'accuracy': best_knn_score,
    'model': best_knn_model,
    'params': best_knn_params
}
print(f"Best KNN Accuracy: {best_knn_score:.4f}")

# Summary
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (method, data) in enumerate(sorted_results, 1):
    print(f"{i}. {method}: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
    print(f"   Best Parameters: {data['params']}")

# Save all models
print("\nSaving models...")
for method, data in results.items():
    model_name = method.lower().replace(' ', '_')
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(data['model'], f)
    with open(f"{model_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved {method} model")

print("\nAll models saved successfully!")
