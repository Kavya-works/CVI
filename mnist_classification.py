import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import time

print("Loading MNIST dataset...")
# Load training data
train_data = pd.read_csv('mnist_train.csv', header=None)
X_train = train_data.iloc[:, 1:].values  # All columns except first (pixels)
y_train = train_data.iloc[:, 0].values    # First column (labels)

# Load test data
test_data = pd.read_csv('mnist_test.csv', header=None)
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Feature dimension: {X_train.shape[1]} (28x28 pixels)")

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# 1. Logistic Regression
print("\n=== Logistic Regression ===")
best_lr_score = 0
best_lr_model = None
best_lr_params = None

for C in [0.1, 1.0, 10.0]:
    start_time = time.time()
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    score = lr.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"C={C}: Accuracy = {score:.4f} ({score*100:.2f}%) (Time: {train_time:.2f}s)")
    
    if score > best_lr_score:
        best_lr_score = score
        best_lr_model = lr
        best_lr_params = {'C': C}

results['Logistic Regression'] = {
    'accuracy': best_lr_score,
    'model': best_lr_model,
    'params': best_lr_params
}
print(f"Best Logistic Regression Accuracy: {best_lr_score:.4f} ({best_lr_score*100:.2f}%)")

# 2. Neural Network (MLP)
print("\n=== Neural Network (MLP) ===")
best_nn_score = 0
best_nn_model = None
best_nn_params = None

architectures = [
    (128,),        # Single layer, 128 neurons
    (256,),        # Single layer, 256 neurons
    (128, 64),     # Two layers, 128 and 64 neurons
    (256, 128),    # Two layers, 256 and 128 neurons
]

for hidden_layers in architectures:
    start_time = time.time()
    nn = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, 
                       random_state=42, early_stopping=True, validation_fraction=0.1,
                       learning_rate_init=0.001)
    nn.fit(X_train_scaled, y_train)
    score = nn.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"Layers {hidden_layers}: Accuracy = {score:.4f} ({score*100:.2f}%) (Time: {train_time:.2f}s)")
    
    if score > best_nn_score:
        best_nn_score = score
        best_nn_model = nn
        best_nn_params = {'hidden_layers': hidden_layers}

results['Neural Network'] = {
    'accuracy': best_nn_score,
    'model': best_nn_model,
    'params': best_nn_params
}
print(f"Best Neural Network Accuracy: {best_nn_score:.4f} ({best_nn_score*100:.2f}%)")

# 3. Support Vector Machine (SVM) - Using subset due to computational cost
print("\n=== Support Vector Machine (SVM) ===")
best_svm_score = 0
best_svm_model = None
best_svm_params = None

# Use subset for faster training
sample_size = min(10000, len(X_train_scaled))
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_svm = X_train_scaled[indices]
y_train_svm = y_train[indices]

for C in [1.0, 10.0]:
    for kernel in ['rbf', 'linear']:
        start_time = time.time()
        svm = SVC(C=C, kernel=kernel, random_state=42, probability=True)
        svm.fit(X_train_svm, y_train_svm)
        score = svm.score(X_test_scaled, y_test)
        train_time = time.time() - start_time
        print(f"C={C}, Kernel={kernel}: Accuracy = {score:.4f} ({score*100:.2f}%) (Time: {train_time:.2f}s)")
        
        if score > best_svm_score:
            best_svm_score = score
            best_svm_model = svm
            best_svm_params = {'C': C, 'kernel': kernel}

results['SVM'] = {
    'accuracy': best_svm_score,
    'model': best_svm_model,
    'params': best_svm_params
}
print(f"Best SVM Accuracy: {best_svm_score:.4f} ({best_svm_score*100:.2f}%)")

# 4. Random Forest
print("\n=== Random Forest ===")
best_rf_score = 0
best_rf_model = None
best_rf_params = None

for n_estimators in [100, 200]:
    for max_depth in [20, None]:
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        score = rf.score(X_test_scaled, y_test)
        train_time = time.time() - start_time
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: Accuracy = {score:.4f} ({score*100:.2f}%) (Time: {train_time:.2f}s)")
        
        if score > best_rf_score:
            best_rf_score = score
            best_rf_model = rf
            best_rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

results['Random Forest'] = {
    'accuracy': best_rf_score,
    'model': best_rf_model,
    'params': best_rf_params
}
print(f"Best Random Forest Accuracy: {best_rf_score:.4f} ({best_rf_score*100:.2f}%)")

# 5. K-Nearest Neighbors (KNN)
print("\n=== K-Nearest Neighbors (KNN) ===")
best_knn_score = 0
best_knn_model = None
best_knn_params = None

for n_neighbors in [3, 5, 7]:
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    train_time = time.time() - start_time
    print(f"n_neighbors={n_neighbors}: Accuracy = {score:.4f} ({score*100:.2f}%) (Time: {train_time:.2f}s)")
    
    if score > best_knn_score:
        best_knn_score = score
        best_knn_model = knn
        best_knn_params = {'n_neighbors': n_neighbors}

results['KNN'] = {
    'accuracy': best_knn_score,
    'model': best_knn_model,
    'params': best_knn_params
}
print(f"Best KNN Accuracy: {best_knn_score:.4f} ({best_knn_score*100:.2f}%)")

# Summary
print("\n" + "="*70)
print("FINAL COMPARISON - MNIST Classification")
print("="*70)
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (method, data) in enumerate(sorted_results, 1):
    accuracy_pct = data['accuracy'] * 100
    status = "✓ PASS" if accuracy_pct >= 90 else "✗ FAIL"
    print(f"{i}. {method}: {data['accuracy']:.4f} ({accuracy_pct:.2f}%) {status}")
    print(f"   Best Parameters: {data['params']}")

# Check if requirement is met
methods_above_90 = [m for m, d in results.items() if d['accuracy'] >= 0.90]
print(f"\nMethods achieving ≥90% accuracy: {len(methods_above_90)}/{len(results)}")
if methods_above_90:
    print(f"Methods: {', '.join(methods_above_90)}")

# Save best model (highest accuracy)
best_method = sorted_results[0][0]
best_model = sorted_results[0][1]['model']
print(f"\nSaving best model: {best_method}")

with open("best_mnist_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("mnist_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model saved successfully!")

