import os
import cv2
import numpy as np
import pickle

def load_model(model_name="neural_network"):
    """Load a trained model and its scaler"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{model_name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_image(model, scaler, image_path, img_size=(64, 64)):
    """Predict class for a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    img_resized = cv2.resize(img, img_size)
    img_flattened = img_resized.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    
    prediction = model.predict(img_scaled)[0]
    probability = model.predict_proba(img_scaled)[0]
    
    return prediction, probability

def test_model(model_name, model, scaler, test_folder, label_name):
    """Test model on a folder of images"""
    print(f"\nTesting {model_name} on {label_name} images:")
    correct = 0
    total = 0
    
    for img_file in os.listdir(test_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_folder, img_file)
            pred, prob = predict_image(model, scaler, img_path)
            if pred is not None:
                result = "Cat" if pred == 0 else "Dog"
                expected = "Cat" if label_name == "Cat" else "Dog"
                is_correct = (pred == 0 and label_name == "Cat") or (pred == 1 and label_name == "Dog")
                correct += is_correct
                total += 1
                status = "✓" if is_correct else "✗"
                print(f"  {status} {img_file}: {result} ({prob[pred]:.2%})")
    
    accuracy = correct / total if total > 0 else 0
    print(f"  Accuracy: {correct}/{total} ({accuracy:.2%})")
    return correct, total

# Try to load models in order of preference
models_to_try = [
    ("neural_network", "Neural Network"),
    ("logistic_regression", "Logistic Regression"),
    ("svm", "SVM"),
    ("random_forest", "Random Forest"),
    ("knn", "KNN"),
]

model = None
scaler = None
model_name_display = None

for model_name, display_name in models_to_try:
    try:
        model, scaler = load_model(model_name)
        model_name_display = display_name
        print(f"Using {display_name} model")
        break
    except FileNotFoundError:
        continue

if model is None:
    print("Error: No trained model found!")
    exit(1)

# Test on test images
base_path = os.path.dirname(os.path.abspath(__file__))
test_cat_folder = os.path.join(base_path, "test", "Cat")
test_dog_folder = os.path.join(base_path, "test", "Dog")

cat_correct, cat_total = test_model(model_name_display, model, scaler, test_cat_folder, "Cat")
dog_correct, dog_total = test_model(model_name_display, model, scaler, test_dog_folder, "Dog")

overall_accuracy = (cat_correct + dog_correct) / (cat_total + dog_total)
print(f"\n{'='*60}")
print(f"Overall Test Accuracy: {cat_correct + dog_correct}/{cat_total + dog_total} ({overall_accuracy:.2%})")
print(f"{'='*60}")

# Test on internet images if folder exists
internet_folder = os.path.join(base_path, "internet_images")
if os.path.exists(internet_folder):
    print("\n" + "="*60)
    print("Testing on Internet Images")
    print("="*60)
    
    internet_images = [f for f in os.listdir(internet_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if internet_images:
        print(f"\nFound {len(internet_images)} image(s) in internet_images folder:")
        for img_file in internet_images:
            img_path = os.path.join(internet_folder, img_file)
            pred, prob = predict_image(model, scaler, img_path)
            if pred is not None:
                result = "Cat" if pred == 0 else "Dog"
                print(f"  {img_file}: {result} ({prob[pred]:.2%})")
    else:
        print("No images found in internet_images folder.")
else:
    print("\nNote: 'internet_images' folder not found.")
    print("To test on internet images, create a folder named 'internet_images' and add your images there.")
