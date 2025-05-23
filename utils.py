import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_image(path, size=(256, 256)):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to load {path}")
        return None
    img = cv2.resize(img, size)  # Resize to fixed shape
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def save_image(img, path):
    img = (img * 255).clip(0, 255).astype('uint8')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def calculate_metrics(pred, gt):
    # Resize images to 7x7 if they are smaller than that
    min_dim = 7
    if pred.shape[0] < min_dim or pred.shape[1] < min_dim:
        pred = cv2.resize(pred, (min_dim, min_dim))
        gt = cv2.resize(gt, (min_dim, min_dim))

    # Clip images to the [0, 1] range
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    
    # Calculate metrics
    mse_val = mean_squared_error(gt.flatten(), pred.flatten())
    mae_val = mean_absolute_error(gt.flatten(), pred.flatten())
    psnr_val = psnr(gt, pred, data_range=1.0)
    
    # Use a win_size of 3 (odd number) for SSIM calculation
    ssim_val = ssim(gt, pred, multichannel=True, data_range=1.0, win_size=3)

    return psnr_val, ssim_val, mse_val, mae_val
