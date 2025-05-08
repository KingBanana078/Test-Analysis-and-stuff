from skimage.metrics import structural_similarity as ssim
import cv2

def compare_ssim(imageA, imageB):
    # Ensure the images have the same size
    assert imageA.shape == imageB.shape, "Images must be the same size."

    # Compute SSIM between two images
    score, diff = ssim(imageA, imageB, full=True)
    return score


# Load images
imageA = cv2.imread('image1.webp', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('image2.webp', cv2.IMREAD_GRAYSCALE)

# Compute SSIM
ssim_score = compare_ssim(imageA, imageB)
print(f"SSIM Score: {ssim_score}")