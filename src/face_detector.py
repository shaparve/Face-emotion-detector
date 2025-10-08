import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import time


class LightFaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=0
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        results = self.face_detection.process(image_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                faces.append((x, y, width, height))

        return faces

    def benchmark_detection(self, image: np.ndarray, num_iterations: int = 100) -> float:
        total_time = 0
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.detect_faces(image)
            total_time += time.perf_counter() - start

        return (total_time / num_iterations) * 1000

    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image_copy

    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()