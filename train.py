import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


def load_data(data_dir):
    images, labels = [], []
    for label in range(43):
        label_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(label_dir):
            continue
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            try:
                img = Image.open(img_path)
                img = img.resize((50, 50))
                images.append(np.array(img))
                labels.append(label)
            except Exception:
                continue
    images = np.array(images) / 255.0
    labels = to_categorical(np.array(labels), num_classes=43)
    return train_test_split(images, labels, test_size=0.2, random_state=42)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    x_train, x_val, y_train, y_val = load_data(args.data_dir)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    model = build_model()
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train traffic sign classifier')
    parser.add_argument('--data-dir', required=True, help='Path to GTSRB training directory')
    parser.add_argument('--output', default='traffic_sign_model.h5', help='Output model file')
    parser.add_argument('--epochs', type=int, default=15)
    main(parser.parse_args())
