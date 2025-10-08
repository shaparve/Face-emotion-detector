import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import time
from typing import Optional, Tuple
import os


class EmotionInference:
    def __init__(self, model_path: str, model_type: str = 'tflite'):
        self.model_type = model_type
        self.model_path = model_path
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.interpreter = None
        self.session = None

        if model_type == 'tflite':
            self._load_tflite_model()
        elif model_type == 'onnx':
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_tflite_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_onnx_model(self):
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=-1)
        face_batch = np.expand_dims(face_expanded, axis=0)

        if self.model_type == 'tflite' and self.input_details[0]['dtype'] == np.uint8:
            face_batch = (face_batch * 255).astype(np.uint8)

        return face_batch

    def predict_emotion(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        preprocessed = self.preprocess_face(face_image)

        if self.model_type == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            if self.output_details[0]['dtype'] == np.uint8:
                predictions = predictions.astype(np.float32) / 255.0

        elif self.model_type == 'onnx':
            predictions = self.session.run([self.output_name], {self.input_name: preprocessed})[0][0]

        predictions = predictions / np.sum(predictions)
        emotion_idx = np.argmax(predictions)
        emotion_label = self.emotion_labels[emotion_idx]
        confidence = predictions[emotion_idx]

        return emotion_label, confidence, predictions

    def benchmark_inference(self, face_image: np.ndarray, num_iterations: int = 100) -> dict:
        preprocessed = self.preprocess_face(face_image)

        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()

            if self.model_type == 'tflite':
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
                self.interpreter.invoke()
                _ = self.interpreter.get_tensor(self.output_details[0]['index'])
            elif self.model_type == 'onnx':
                _ = self.session.run([self.output_name], {self.input_name: preprocessed})

            latencies.append((time.perf_counter() - start) * 1000)

        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies)
        }

    def get_model_size(self) -> float:
        if os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0


class SeatbeltDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model()

    def create_simple_model(self, input_shape=(64, 64, 3)):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def get_driver_roi(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        roi_x = int(w * 0.1)
        roi_y = int(h * 0.3)
        roi_w = int(w * 0.4)
        roi_h = int(h * 0.5)

        roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        return roi

    def detect_seatbelt_heuristic(self, image: np.ndarray) -> bool:
        roi = self.get_driver_roi(image)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        lines = cv2.HoughLinesP(mask, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            diagonal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 20 < angle < 70:
                    diagonal_lines += 1

            return diagonal_lines >= 2

        return False

    def load_model(self):
        if self.model_path:
            self.model = tf.keras.models.load_model(self.model_path)

    def save_model(self, path: str):
        if self.model:
            self.model.save(path)