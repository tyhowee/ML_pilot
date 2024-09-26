import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Step 1: Load the Image
image_path = '/Users/thowe/Desktop/MAP3_adjusted.tif'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Select ROIs using OpenCV's selectROIs method
print("Select multiple ROIs and press Enter. Press ESC when done.")
rois = cv2.selectROIs("Select ROIs", image, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# Initialize storage for ROI data
roi_data = []
labels = []

# Step 3: Extract pixels from selected ROIs
for i, roi in enumerate(rois):
    x, y, w, h = roi
    roi_image = image[y:y + h, x:x + w]
    roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    roi_data.append(roi_image_rgb.reshape(-1, 3))  # Flatten to (number_of_pixels, 3)
    labels.extend([i] * roi_image_rgb.size // 3)  # Assign a label to each pixel in ROI

# Combine all ROI data into training arrays
X_train = np.vstack(roi_data)
y_train = np.array(labels)

# Step 4: Train a k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Prepare the Full Image for Classification
image_flattened = image_rgb.reshape(-1, 3)  # Flatten the image to (number_of_pixels, 3)
predicted_labels = knn.predict(image_flattened)
classified_image = predicted_labels.reshape(image_rgb.shape[:2])  # Reshape to original image shape

# Step 6: Display the Classification Results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(classified_image, cmap='tab10')
plt.title("Classified Image")

plt.show()