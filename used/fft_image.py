import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/home/jarvis/CS/DIP/OSDetector/data/train/images/img_0002.jpg"

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

image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
image_filtered_normalized = cv2.normalize(image_filtered, None, 0, 1, cv2.NORM_MINMAX)

plt.subplot(1, 2, 1)
plt.imshow(image_normalized, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
cmap_filtered = plt.imshow(image_filtered_normalized, cmap='gray')
plt.title("Filtered Image")


plt.show()
