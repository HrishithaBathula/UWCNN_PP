
import os
import numpy as np
from model import uwcnn_pp
from utils import load_image, save_image, calculate_metrics

# Define directories
input_dir = 'dataset/test_images'  # CHANGED: now uses test_images/
gt_dir = 'dataset/ground_truth'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Load model and weights
model = uwcnn_pp(input_shape=(None, None, 3))
model.load_weights('uwcnn_pp_best.keras')

# Prepare metrics storage
psnr_list, ssim_list, mse_list, mae_list = [], [], [], []
metrics_table = []
metrics_table.append(f"{'Image':<25} {'PSNR':>8} {'SSIM':>8} {'MSE':>12} {'MAE':>12}")

# Evaluation loop
for fname in os.listdir(input_dir):
    inp_path = os.path.join(input_dir, fname)
    gt_path = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth not found for {fname}. Skipping.")
        continue

    inp = load_image(inp_path)
    gt = load_image(gt_path)

    if inp is None or gt is None:
        print(f"Warning: Could not load {fname}. Skipping.")
        continue

    # Predict and save
    pred = model.predict(np.expand_dims(inp, axis=0))[0]
    save_image(pred, os.path.join(output_dir, fname))

    # Evaluate
    psnr, ssim, mse, mae = calculate_metrics(pred, gt)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    mse_list.append(mse)
    mae_list.append(mae)

    print(f"{fname}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, MSE={mse:.5f}, MAE={mae:.5f}")
    metrics_table.append(f"{fname:<25} {psnr:8.2f} {ssim:8.4f} {mse:12.5f} {mae:12.5f}")

# Averages
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)

print("\nAverage Metrics:")
print(f"PSNR: {avg_psnr:.2f}")
print(f"SSIM: {avg_ssim:.4f}")
print(f"MSE : {avg_mse:.5f}")
print(f"MAE : {avg_mae:.5f}")

metrics_table.append("-" * 65)
metrics_table.append(f"{'Average':<25} {avg_psnr:8.2f} {avg_ssim:8.4f} {avg_mse:12.5f} {avg_mae:12.5f}")

# Save metrics to file
with open("metrics_eval.txt", "w") as f:
    f.write("\n".join(metrics_table))
