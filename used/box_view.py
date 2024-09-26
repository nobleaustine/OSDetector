import cv2
import json
import os

image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/images/img_0097.jpg"
output_json_path = "/home/jarvis/CS/DIP/OSDetector/data/train/bounding_boxes.json"

image = cv2.imread(image_path)

image_file_name = os.path.basename(image_path)

with open(output_json_path, 'r') as json_file:
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

cv2.imshow("Image with Bounding Box", image)

# Wait for "Enter" key to be pressed (ASCII code 13)
while True:
    key = cv2.waitKey(1)  # Check for a key press
    if key == 13:  # 13 is the ASCII code for the Enter key
        break

cv2.destroyAllWindows()

# Optionally, save the image with the bounding box
# output_path = "output_image.jpg"
# cv2.imwrite(output_path, image)