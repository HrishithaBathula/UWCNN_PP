import os
import numpy as np
from model import uwcnn_pp
from utils import load_image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

input_dir = 'dataset/input'
gt_dir = 'dataset/ground_truth'

def load_dataset():
    x, y = [], []
    for fname in os.listdir(input_dir):
        inp_path = os.path.join(input_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"Warning: Missing ground truth for {fname}. Skipping.")
            continue
        inp = load_image(inp_path)
        gt = load_image(gt_path)
        if inp is None or gt is None:
            continue  # Skip if loading failed
        x.append(inp)
        y.append(gt)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    x_train, y_train = load_dataset()
    model = uwcnn_pp(input_shape=(x_train.shape[1], x_train.shape[2], 3))
    model.compile(optimizer=Adam(1e-4), loss='mse')

    model.summary()
    checkpoint = ModelCheckpoint('uwcnn_pp_best.keras', monitor='loss', save_best_only=True)
    model.fit(x_train, y_train, batch_size=2, epochs=300, callbacks=[checkpoint])
