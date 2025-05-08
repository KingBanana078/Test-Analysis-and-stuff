"""from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

def compare_ssim(imageA, imageB):
    assert imageA.shape == imageB.shape, "Images must be the same size."
    score, diff = ssim(imageA, imageB, full=True)
    return score

# Load and convert to grayscale using Pillow
imageA = np.array(Image.open('image1.webp').convert('L'))
imageB = np.array(Image.open('image2.webp').convert('L'))

# Compute SSIM
ssim_score = compare_ssim(imageA, imageB)
print(f"SSIM Score: {ssim_score}")"""


"""import io
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load Deep Mantle
with open(r"C:\Users\julia\OneDrive\Pictures\Screenshots\deep_mantle.png", "rb") as file:
    deep_mantle = Image.open(io.BytesIO(file.read()))

# Load Asthenosphere
with open(r"C:\Users\julia\OneDrive\Pictures\Screenshots\Asthenosphere.png", "rb") as file:
    asthenosphere = Image.open(io.BytesIO(file.read()))

# Load Magma Ocean
with open(r"C:\Users\julia\OneDrive\Pictures\Screenshots\Magma_ocean.png", "rb") as file:
    magma_ocean = Image.open(io.BytesIO(file.read()))

# Convert to grayscale
img1 = magma_ocean.convert('L')
img2 = deep_mantle.convert('L')

# Resize to match dimensions
img2 = img2.resize(img1.size)

# Convert to NumPy arrays
imageA = np.array(img1)
imageB = np.array(img2)

# Compute SSIM
score, diff = ssim(imageA, imageB, full=True)
print(f"SSIM Score: {score}")"""

import io
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load Deep Mantle
with open(r"C:\Users\julia\OneDrive\Dokument\deep_mantle.jpg", "rb") as file:
    deep_mantle = Image.open(io.BytesIO(file.read()))

# Load Asthenosphere
with open(r"C:\Users\julia\OneDrive\Dokument\asthentosphere.jpg", "rb") as file:
    asthenosphere = Image.open(io.BytesIO(file.read()))

# Load Magma Ocean
with open(r"C:\Users\julia\OneDrive\Dokument\magma_ocean.jpg", "rb") as file:
    magma_ocean = Image.open(io.BytesIO(file.read()))

with open(r"C:\Users\julia\OneDrive\Pictures\Screenshots\test2.png", "rb") as file:
    test = Image.open(io.BytesIO(file.read()))

# Convert to grayscale
img1 = test.convert('L')
img2 = asthenosphere.convert('L')


# Resize to match dimensions
img2 = img2.resize(img1.size)


# Convert to NumPy arrays
imageA = np.array(img1)
imageB = np.array(img2)

# Compute SSIM
score, diff = ssim(imageA, imageB, full=True)
print(f"SSIM Score: {score}")

