
import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(inputs, filters=64, name_prefix="res"):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name_prefix}_conv1")(inputs)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name_prefix}_conv2")(x)
    x = layers.Conv2D(filters, 3, padding='same', name=f"{name_prefix}_conv3")(x)  # Keep filters=64
    out = layers.Add(name=f"{name_prefix}_add")([inputs, x])
    return out

def uwcnn_pp(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape, name='input_image')

    # Initial feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='init_conv1')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='init_conv2')(x)

    # Stack of residual blocks
    for i in range(5):  # UWCNN++ typically uses more blocks than UWCNN
        x = residual_block(x, filters=64, name_prefix=f"res_block_{i+1}")

    # Final reconstruction layer
    output = layers.Conv2D(3, 3, padding='same', name='final_output_conv')(x)
    output = layers.Activation('relu', name='final_relu')(output)

    model = models.Model(inputs=inputs, outputs=output, name="UWCNN_PP")
    return model
