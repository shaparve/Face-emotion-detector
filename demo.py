import cv2
import numpy as np
import argparse
from src.face_detector import LightFaceDetector
from src.inference import EmotionInference, SeatbeltDetector
import os


def run_realtime_demo(model_path: str, model_type: str = 'tflite',
                     enable_seatbelt: bool = False):
    face_detector = LightFaceDetector(min_detection_confidence=0.5)
    emotion_inference = EmotionInference(model_path, model_type)

    seatbelt_detector = None
    if enable_seatbelt:
        seatbelt_detector = SeatbeltDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Press 'q' to quit")
    print("Press 's' to save screenshot")

    frame_count = 0
    total_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            emotion, confidence, probs = emotion_inference.predict_emotion(face_roi)

            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bar_height = 100
            bar_width = 15
            bar_x = x + w + 10
            for i, (emotion_name, prob) in enumerate(zip(emotion_inference.emotion_labels, probs)):
                bar_y = y + i * 15
                bar_length = int(prob * 100)
                cv2.rectangle(frame, (bar_x, bar_y),
                            (bar_x + bar_length, bar_y + 10),
                            (0, 255, 0), -1)
                cv2.putText(frame, f"{emotion_name[:3]}", (bar_x + 105, bar_y + 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if enable_seatbelt and seatbelt_detector:
            seatbelt_detected = seatbelt_detector.detect_seatbelt_heuristic(frame)
            status_text = "Seatbelt: " + ("Yes" if seatbelt_detected else "No")
            color = (0, 255, 0) if seatbelt_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_text = f"Faces: {len(faces)}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Emotion Detection Demo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'screenshot_{frame_count}.png', frame)
            print(f"Screenshot saved: screenshot_{frame_count}.png")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def run_image_demo(image_path: str, model_path: str, model_type: str = 'tflite',
                   enable_seatbelt: bool = False):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    face_detector = LightFaceDetector(min_detection_confidence=0.5)
    emotion_inference = EmotionInference(model_path, model_type)

    image = cv2.imread(image_path)
    original = image.copy()

    faces = face_detector.detect_faces(image)
    print(f"Detected {len(faces)} face(s)")

    face_benchmark = None
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = original[y:y+h, x:x+w]
        emotion, confidence, probs = emotion_inference.predict_emotion(face_roi)

        if i == 0:
            face_benchmark = emotion_inference.benchmark_inference(face_roi, num_iterations=50)

        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(image, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"\nFace {i+1}:")
        print(f"  Predicted: {emotion} ({confidence:.2%})")
        print("  All probabilities:")
        for emotion_name, prob in zip(emotion_inference.emotion_labels, probs):
            print(f"    {emotion_name}: {prob:.4f}")

    if face_benchmark:
        print(f"\nBenchmark Results:")
        print(f"  Mean latency: {face_benchmark['mean_latency_ms']:.2f} ms")
        print(f"  Std latency: {face_benchmark['std_latency_ms']:.2f} ms")

    model_size = emotion_inference.get_model_size()
    print(f"\nModel size: {model_size:.2f} MB")

    if enable_seatbelt:
        seatbelt_detector = SeatbeltDetector()
        seatbelt_detected = seatbelt_detector.detect_seatbelt_heuristic(original)
        print(f"\nSeatbelt detection: {'Yes' if seatbelt_detected else 'No'}")

    cv2.imshow('Emotion Detection Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, image)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Detection Demo')
    parser.add_argument('--mode', type=str, choices=['realtime', 'image'],
                       default='realtime', help='Demo mode')
    parser.add_argument('--image', type=str, help='Path to input image (for image mode)')
    parser.add_argument('--model', type=str, default='models/emotion_classifier_int8.tflite',
                       help='Path to emotion model')
    parser.add_argument('--model_type', type=str, choices=['tflite', 'onnx'],
                       default='tflite', help='Model type')
    parser.add_argument('--seatbelt', action='store_true',
                       help='Enable seatbelt detection')

    args = parser.parse_args()

    if args.mode == 'realtime':
        run_realtime_demo(args.model, args.model_type, args.seatbelt)
    elif args.mode == 'image':
        if not args.image:
            print("Error: Please provide image path with --image argument")
        else:
            run_image_demo(args.image, args.model, args.model_type, args.seatbelt)