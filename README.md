# HistogramEqualization
Berikut adalah kodingan untuk Algoritma Histogram Equalization :
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import BytesIO
from PIL import Image

# Step 1: Upload the image
uploaded = files.upload()
if not uploaded:
    print("No file uploaded.")
    exit()

# Load the uploaded image in RGB format
file_name = list(uploaded.keys())[0]
image = Image.open(BytesIO(uploaded[file_name])).convert('RGB')
image = np.array(image)

# Get image dimensions
height, width, _ = image.shape

# Step 2: Convert RGB to grayscale manually
grayscale_image = np.zeros((height, width), dtype=int)
for i in range(width):
    for j in range(height):
        r, g, b = image[j, i]
        grayscale_value = int((r + g + b) / 3)  # Calculate grayscale as average of RGB
        grayscale_image[j, i] = grayscale_value

# Step 3: Compute the histogram for the grayscale image
histogram = np.zeros(256, dtype=int)
for i in range(width):
    for j in range(height):
        pixel_value = grayscale_image[j, i]
        histogram[pixel_value] += 1

# Step 4: Calculate cumulative distribution function (CDF)
cdf = np.cumsum(histogram)
cdf_min = cdf[cdf > 0][0]  # Minimum non-zero value in CDF

# Step 5: Histogram equalization mapping
equalized_grayscale = np.zeros_like(grayscale_image)
for i in range(width):
    for j in range(height):
        pixel_value = grayscale_image[j, i]
        equalized_grayscale[j, i] = np.round(255 * (cdf[pixel_value] - cdf_min) / (height * width - cdf_min)).astype(np.uint8)

# Step 6: Apply the equalized grayscale values to the RGB image
equalized_image = np.zeros_like(image)
for i in range(width):
    for j in range(height):
        equalized_value = equalized_grayscale[j, i]
        equalized_image[j, i] = [equalized_value, equalized_value, equalized_value]  # Set RGB channels to equalized grayscale

# Step 7: Display results
# Original and equalized histograms
original_histogram, bins = np.histogram(grayscale_image.flatten(), bins=256, range=[0, 256])
equalized_histogram, bins = np.histogram(equalized_grayscale.flatten(), bins=256, range=[0, 256])

# Plot original and equalized images
plt.figure(figsize=(12, 6))

# Original image and histogram
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image (RGB)')
plt.subplot(2, 2, 2)
plt.plot(original_histogram)
plt.title('Original Histogram (Grayscale)')

# Equalized image and histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image)
plt.title('Equalized Image (RGB)')
plt.subplot(2, 2, 4)
plt.plot(equalized_histogram)
plt.title('Equalized Histogram (Grayscale)')

plt.tight_layout()
plt.show()

# Save the result
output_path = "/content/equalized_image_rgb.png"
cv2.imwrite(output_path, cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
print(f"Equalized image saved as: {output_path}")

```
Berikut adalah penjelasan singkat tentang cara kerja kode:

1. **Mengunggah dan Memuat Gambar**: Gambar yang diunggah dibaca dan dikonversi menjadi format RGB menggunakan `PIL` dan diubah menjadi array `numpy`.
2. **Konversi ke Grayscale**: Gambar RGB diubah menjadi grayscale dengan menghitung rata-rata nilai RGB setiap piksel.
3. **Menghitung Histogram**: Histogram intensitas piksel pada gambar grayscale dihitung.
4. **Menghitung CDF**: Fungsi distribusi kumulatif (CDF) dihitung berdasarkan histogram untuk membantu distribusi intensitas yang lebih merata.
5. **Peningkatan Histogram**: Nilai piksel pada gambar grayscale dipetakan ulang berdasarkan CDF untuk meningkatkan kontras.
6. **Menerapkan ke Gambar RGB**: Nilai grayscale yang telah ditingkatkan diterapkan ke gambar RGB, menjadikannya gambar grayscale dengan kontras yang lebih baik.
7. **Menampilkan dan Menyimpan Hasil**: Gambar asli dan yang telah disesuaikan ditampilkan bersama histogramnya, dan gambar yang telah disesuaikan disimpan.

Secara keseluruhan, kode ini melakukan **peningkatan kontras gambar** menggunakan teknik histogram equalization.
