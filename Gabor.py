import cv2
import numpy as np
import torch  # PyTorch for GPU acceleration
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import os
import joblib
from glob import glob

# ✅ **Check if cuML is installed (for GPU-accelerated SVM)**
try:
    from cuml.svm import SVC  # GPU-based SVM
    from cuml.model_selection import train_test_split
    use_gpu = True
except ImportError:
    from sklearn.svm import SVC  # CPU-based SVM fallback
    from sklearn.model_selection import train_test_split
    use_gpu = False

# ✅ **Check if OpenCV has CUDA enabled**
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

# ✅ **Dataset Path**
DATASET_PATH = "F:/clothfoldingmachine/my_dataset"

# ✅ **Load images from dataset**
def load_images(folder_path):
    image_files = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.jpeg"))  
    if not image_files:
        print(f"❌ No images found in '{folder_path}'. Check the path.")
        return []

    print(f"✅ Found {len(image_files)} images.")
    images = [(cv2.imread(img), os.path.basename(img).split('_')[0]) for img in image_files]
    return images

# ✅ **Apply Gabor filter using PyTorch (GPU)**
def apply_gabor_filter(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grayscale = torch.tensor(rgb2gray(image), device=device, dtype=torch.float32)  # Move to GPU/CPU
    gabor_features = []

    for theta in torch.arange(0, torch.pi, torch.pi / 4, device=device):  # Different orientations
        for frequency in (0.1, 0.2, 0.3):  # Different scales
            gabor_response = torch.tensor(
                gabor(grayscale.cpu().numpy(), frequency=frequency, theta=theta.item())[0],
                device=device
            )
            gabor_features.append(gabor_response)

    return torch.stack(gabor_features).mean(dim=(1, 2)).cpu().numpy()  # Convert to NumPy only at the end

# ✅ **Superpixel segmentation using Scikit-Image**
def segment_superpixels(image, n_segments=200):
    return slic(image, n_segments=n_segments, compactness=10, sigma=1, start_label=0)

# ✅ **Edge detection using OpenCV CUDA (with proper fallback)**
def detect_edges(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if cuda_available:
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(grayscale)
            edges = cv2.cuda.Canny(gpu_mat, 50, 150)  # CUDA-accelerated Canny Edge Detection
            return edges.download()  # Move result from GPU to CPU
        except cv2.error:
            print("⚠ OpenCV CUDA failed! Using CPU Canny instead.")

    return canny(rgb2gray(image), sigma=2)  # Use CPU-based Canny as fallback

# ✅ **Feature extraction pipeline**
def extract_features(image):
    print("🟢 Extracting features...")

    gabor_features = apply_gabor_filter(image)
    print("✅ Gabor filter applied")

    superpixel_segments = segment_superpixels(image)
    print("✅ Superpixel segmentation done")

    edge_map = detect_edges(image)
    print("✅ Edge detection completed")

    if gabor_features.size == 0 or superpixel_segments.size == 0 or edge_map.size == 0:
        print("❌ Feature extraction failed: Empty features generated.")
        return None
    
    feature_vector = np.hstack([
        gabor_features,  # Already mean-reduced
        np.array([np.mean(edge_map)]),  # Edge density
        np.array([len(np.unique(superpixel_segments))])  # Number of unique superpixels
    ])
    
    print("✅ Feature vector created")
    return feature_vector

# ✅ **Load dataset and extract features**
def prepare_dataset(image_folder):
    images = load_images(image_folder)
    
    feature_vectors = []
    labels = []
    
    for img, label in images:
        features = extract_features(img)
        if features is not None:
            feature_vectors.append(features)
            labels.append(label)
    
    return np.array(feature_vectors), np.array(labels)

# ✅ **Train and evaluate GPU-accelerated SVM classifier**
def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = SVC(kernel="rbf", C=10, gamma=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"✅ Classification Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, "svm_model.pkl")  # Save the model
    return model

# ✅ **Test Image Visualization**
def visualize_test_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found! Check the path.")
        return
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Failed to read '{image_path}'. Check file format.")
        return

    model = joblib.load("svm_model.pkl")  # Load trained model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract features
    features = extract_features(image)
    if features is None:
        print("❌ Feature extraction failed.")
        return

    # Reshape for prediction
    features = features.reshape(1, -1)

    # Predict label
    predicted_label = model.predict(features)[0]

    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")

    # Edge detection visualization
    edges = detect_edges(image)
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")

    plt.show()

# ✅ **Main execution**
if __name__ == "__main__":
    model_path = "svm_model.pkl"

    # ✅ Check if trained model exists
    if os.path.exists(model_path):
        print("✅ Using existing trained model.")
        model = joblib.load(model_path)  # Load existing model
    else:
        print("🔄 Training new model...")
        features, labels = prepare_dataset(DATASET_PATH)
        model = train_classifier(features, labels)

    # ✅ Test a new image
    test_image_path = "F:/clothfoldingmachine/test_image.jpg"

    # ✅ Check if test image exists
    if os.path.exists(test_image_path):
        visualize_test_image(test_image_path)
    else:
        print(f"❌ Error: Test image '{test_image_path}' not found! Add a test image and try again.")
