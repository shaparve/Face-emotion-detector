import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional
import os


class EmotionClassifier:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def build_lightweight_model(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_shape)

        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='lightweight_emotion_classifier')
        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            self.build_lightweight_model()

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=1, name='top1_accuracy')]
        )

    def train(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def export_tflite(self, output_path: str, quantize: bool = True):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"TFLite model saved to {output_path}")
        print(f"Model size: {model_size:.2f} MB")
        return model_size

    def export_onnx(self, output_path: str):
        import tf2onnx
        spec = (tf.TensorSpec((None, *self.input_shape), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec)
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ONNX model saved to {output_path}")
        print(f"Model size: {model_size:.2f} MB")
        return model_size

    def get_model_parameters(self):
        if self.model:
            total_params = self.model.count_params()
            return total_params
        return 0