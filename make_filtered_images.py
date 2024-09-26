import cv2
import numpy as np
import os

input_dir = "/home/jarvis/CS/DIP/OSDetector/data/train/images/"
output_dir = "/home/jarvis/CS/DIP/OSDetector/data/train/filtered_images/"

os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), np.uint8)
        r = 30
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0

        fshift_filtered = fshift * mask

        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        image_filtered = np.abs(image_filtered)

        image_filtered_normalized = cv2.normalize(image_filtered, None, 0, 1, cv2.NORM_MINMAX)

        filtered_filename = os.path.splitext(filename)[0] + '_filtered.npy'
        output_path = os.path.join(output_dir, filtered_filename)

        np.save(output_path, image_filtered_normalized)
        print(f"Filtered image saved as '{filtered_filename}'.")

print("All images have been processed.")
