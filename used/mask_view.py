import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt

image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/images/img_0002.jpg"
output_json_path = "/home/jarvis/CS/DIP/OSDetector/data/train/bounding_boxes.json"
mask_path = "/home/jarvis/CS/DIP/OSDetector/data/train/prob_masks/prob_mask_img_0002.npy"

image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Get the image file name
image_file_name = os.path.basename(image_path)

# Load the bounding boxes from the JSON file
with open(output_json_path, 'r') as json_file:
    bounding_boxes = json.load(json_file)

# Check if the image file name exists in the bounding boxes
if image_file_name in bounding_boxes and len(bounding_boxes[image_file_name])!=0:  # Check if there are any bounding boxes
    bbox = bounding_boxes[image_file_name][0]  
    x_min = int(bbox['x'] - bbox['width'] / 2)
    y_min = int(bbox['y'] - bbox['height'] / 2)
    x_max = int(bbox['x'] + bbox['width'] / 2)
    y_max = int(bbox['y'] + bbox['height'] / 2)

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Put text label
    label = f"Class: {1}, Confidence: {bbox['confidence']:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print(f"No bounding box found for image: {image_file_name}")

mask = np.load(mask_path)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image)
axs[0].set_title("Image with Bounding Box")

cmap = axs[1].imshow(mask, cmap='viridis')
axs[1].set_title("Probability Mask")

plt.tight_layout()
plt.show()


