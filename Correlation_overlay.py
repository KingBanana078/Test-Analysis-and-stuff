from skimage.metrics import structural_similarity as ssim
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
print(f"SSIM Score: {ssim_score}")