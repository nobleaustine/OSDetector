import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

base_path = "/home/jarvis/CS/DIP/OSDetector/data/train"
image_file_name = "img_0189"
image_path = os.path.join(base_path, "images", f'{image_file_name}.jpg')

bounding_box = os.path.join(base_path,"bounding_boxes.json")
gaussian_mask_path = os.path.join(base_path, "prob_masks",f"prob_mask_{image_file_name}.npy")
filtered_image_path = os.path.join(base_path, "filtered_images", f"{image_file_name}_filtered.npy")
segmentation_label_path = os.path.join(base_path, "labels", f"{image_file_name}.png")


# print("Image Path:", image_path)
# print("Bounding Box Mask Path:", bounding_box)
# print("Gaussian Mask Path:", gaussian_mask_path)
# print("Filtered Image Path:", filtered_image_path)

image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

image_file_name = os.path.basename(image_path)

with open(bounding_box, 'r') as json_file:
    bounding_boxes = json.load(json_file)

if image_file_name in bounding_boxes and len(bounding_boxes[image_file_name]) > 0:
    bbox = bounding_boxes[image_file_name][0]  
    x_min = int(bbox['x'] - bbox['width'] / 2)
    y_min = int(bbox['y'] - bbox['height'] / 2)
    x_max = int(bbox['x'] + bbox['width'] / 2)
    y_max = int(bbox['y'] + bbox['height'] / 2)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    label = f"Class: {1}, Confidence: {bbox['confidence']:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

gaussian_mask = np.load(gaussian_mask_path)

filtered_image = np.load(filtered_image_path)

segmentation_label = cv2.imread(segmentation_label_path)
segmentation_label = cv2.cvtColor(segmentation_label, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))


axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[0].axis('off')

# Display the bounding box mask
# reshaped_array = segmentation_label.reshape(-1, 3)
# print(np.unique(reshaped_array,axis=0))
axs[1].imshow(segmentation_label, cmap='gray')
axs[1].set_title("Segmentation Label")
axs[1].axis('off')

axs[2].imshow(gaussian_mask, cmap='gray')
axs[2].set_title("Gaussian Mask")
axs[2].axis('off')

# Display the filtered image
axs[3].imshow(filtered_image, cmap='gray')
axs[3].set_title("Filtered Image")
axs[3].axis('off')

plt.tight_layout()
plt.show()
