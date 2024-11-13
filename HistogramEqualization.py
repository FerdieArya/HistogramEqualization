import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import BytesIO
from PIL import Image

uploaded = files.upload()
if not uploaded:
    print("No file uploaded.")
    exit()

file_name = list(uploaded.keys())[0]
image = Image.open(BytesIO(uploaded[file_name])).convert('RGB')
image = np.array(image)

height, width, _ = image.shape

grayscale_image = np.zeros((height, width), dtype=int)
for i in range(width):
    for j in range(height):
        r, g, b = image[j, i]
        grayscale_value = int((r + g + b) / 3)  # Calculate grayscale as average of RGB
        grayscale_image[j, i] = grayscale_value

histogram = np.zeros(256, dtype=int)
for i in range(width):
    for j in range(height):
        pixel_value = grayscale_image[j, i]
        histogram[pixel_value] += 1

cdf = np.cumsum(histogram)
cdf_min = cdf[cdf > 0][0]  # Minimum non-zero value in CDF

equalized_grayscale = np.zeros_like(grayscale_image)
for i in range(width):
    for j in range(height):
        pixel_value = grayscale_image[j, i]
        equalized_grayscale[j, i] = np.round(255 * (cdf[pixel_value] - cdf_min) / (height * width - cdf_min)).astype(np.uint8)

equalized_image = np.zeros_like(image)
for i in range(width):
    for j in range(height):
        equalized_value = equalized_grayscale[j, i]
        equalized_image[j, i] = [equalized_value, equalized_value, equalized_value]  # Set RGB channels to equalized grayscale

original_histogram, bins = np.histogram(grayscale_image.flatten(), bins=256, range=[0, 256])
equalized_histogram, bins = np.histogram(equalized_grayscale.flatten(), bins=256, range=[0, 256])

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image (RGB)')
plt.subplot(2, 2, 2)
plt.plot(original_histogram)
plt.title('Original Histogram (Grayscale)')

plt.subplot(2, 2, 3)
plt.imshow(equalized_image)
plt.title('Equalized Image (RGB)')
plt.subplot(2, 2, 4)
plt.plot(equalized_histogram)
plt.title('Equalized Histogram (Grayscale)')

plt.tight_layout()
plt.show()

output_path = "/content/equalized_image_rgb.png"
cv2.imwrite(output_path, cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
print(f"Equalized image saved as: {output_path}")
