import requests
import os
import json
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("ROBOFLOW_API_KEY")
model_id = "-biv3n/1"  
image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/images/img_0001.jpg"

headers = {
    "Authorization": f"Bearer {api_key}"
}

bounding_boxes = {}
confidence_threshold = 0.5  

with open(image_path, "rb") as image_file:
    
    response = requests.post(
        f"https://detect.roboflow.com/{model_id}?api_key={api_key}",
        files={"file": image_file},
        headers=headers
    )

if response.status_code == 200:
    result = response.json()
    # Filter predictions by confidence threshold
    filtered_predictions = [pred for pred in result['predictions'] if pred['confidence'] >= confidence_threshold]
    print(filtered_predictions)

else:
    print(f"Error: {response.status_code}, {response.text}")

