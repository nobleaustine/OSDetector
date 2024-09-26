import cv2
import json
import os
import numpy as np

image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/images/img_0010.jpg"
output_json_path = "/home/jarvis/CS/DIP/OSDetector/data/train/bounding_boxes.json"

image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

image_file_name = os.path.basename(image_path)

with open(output_json_path, 'r') as json_file:
    bounding_boxes = json.load(json_file)

if image_file_name in bounding_boxes:
    bbox = bounding_boxes[image_file_name][0]
    x_min = int(bbox['x'] - bbox['width'] / 2)
    y_min = int(bbox['y'] - bbox['height'] / 2)
    x_max = int(bbox['x'] + bbox['width'] / 2)
    y_max = int(bbox['y'] + bbox['height'] / 2)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    label = f"Class: {1}, Confidence: {bbox['confidence']:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    mask = np.zeros((image_height, image_width), dtype=np.float32)

    center_x = int(bbox['x'])
    center_y = int(bbox['y'])

    sigma_x = bbox['width'] / 2  
    sigma_y = bbox['height'] / 2 


    x = np.arange(0, image_width)
    y = np.arange(0, image_height)
    x, y = np.meshgrid(x, y)

    mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
    mask = (mask / np.max(mask) * 255).astype(np.uint8)
    
    gaussian_colormap = cv2.applyColorMap(mask, cv2.COLORMAP_VIRIDIS)
    combined_image = np.hstack((image, gaussian_colormap))

    cv2.imshow("Image with Bounding Box and Gaussian Mask", combined_image)

while True:
    key = cv2.waitKey(1)  
    if key == 13:  
        break

cv2.destroyAllWindows()
# Optionally, save the image with the bounding box and Gaussian mask
# output_path = "output_image.jpg"



