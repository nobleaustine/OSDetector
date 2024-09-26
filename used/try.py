import cv2
import numpy as np

# Path to the PNG image
image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/labels_1D/img_0008.png"  # Change this to your actual image path

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to load the image as is

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Print image shape
    print(f"Image shape: {image.shape}")
    
    # If the image is in color (3 channels), convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get unique elements
    unique_elements = np.unique(image)

    # Display the unique elements
    print("Unique elements in the image:", unique_elements)
