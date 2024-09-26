import requests
import os
import json
from dotenv import load_dotenv
import os

load_dotenv() 
api_key = os.getenv("ROBOFLOW_API_KEY")
model_id = "-biv3n/1"  
image_folder = "/home/jarvis/CS/DIP/OSDetector/data/train/images/"
output_json_path = "/home/jarvis/CS/DIP/OSDetector/data/train/bounding_boxes.json"

headers = {
    "Authorization": f"Bearer {api_key}"
}


bounding_boxes = {}

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

for image_file_name in image_files:
    image_path = os.path.join(image_folder, image_file_name)
    
    with open(image_path, "rb") as image_file:
        
        response = requests.post(
            f"https://detect.roboflow.com/{model_id}?api_key={api_key}",
            files={"file": image_file},
            headers=headers
        )

    if response.status_code == 200:
        result = response.json()
        bounding_boxes[image_file_name] = result['predictions']
        print(f"Processed {image_file_name}")

    else:
        print(f"Error: {response.status_code}, {response.text}")

with open(output_json_path, 'w') as json_file:
    json.dump(bounding_boxes, json_file, indent=4)

print(f"Bounding box details saved to {output_json_path}")
