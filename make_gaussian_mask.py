import cv2
import json
import os
import numpy as np


image_folder = "/home/jarvis/CS/DIP/OSDetector/data/train/images/"
output_json_path = "/home/jarvis/CS/DIP/OSDetector/data/train/bounding_boxes.json"
output_mask_folder = "/home/jarvis/CS/DIP/OSDetector/data/train/prob_masks/"
epsilon = 1e-6


os.makedirs(output_mask_folder, exist_ok=True)

with open(output_json_path, 'r') as json_file:
    bounding_boxes = json.load(json_file)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

for image_file_name in image_files:
    image_path = os.path.join(image_folder, image_file_name)
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    if image_file_name in bounding_boxes:

        if bounding_boxes[image_file_name]:
            bbox = bounding_boxes[image_file_name][0] 
            mask = np.zeros((image_height, image_width), dtype=np.float32)

            center_x = int(bbox['x'])
            center_y = int(bbox['y'])

            sigma_x = bbox['width'] / 2  
            sigma_y = bbox['height'] / 2  

            x = np.arange(0, image_width)
            y = np.arange(0, image_height)
            x, y = np.meshgrid(x, y)

        
            mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
        else:
            
            mask = np.full((image_height, image_width), epsilon, dtype=np.float32)
    else:
        mask = np.full((image_height, image_width), epsilon, dtype=np.float32)

    prob_mask_output_path = os.path.join(output_mask_folder, f"prob_mask_{os.path.splitext(image_file_name)[0]}.npy")
    np.save(prob_mask_output_path, mask)

    print(f"Processed and saved probability mask for {image_file_name}")

print("All images processed.")
