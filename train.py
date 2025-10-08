import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from src.data_loader import FER2013DataLoader
from src.emotion_classifier import EmotionClassifier
from src.evaluation import ModelEvaluator
from src.inference import EmotionInference
import warnings
warnings.filterwarnings('ignore')


def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    print("=" * 50)
    print("Face Emotion Detection Training Pipeline")
    print("=" * 50)

    data_loader = FER2013DataLoader(data_dir=args.data_dir)
    print("\n1. Loading FER2013 dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.prepare_train_val_test(
        test_size=0.2, val_size=0.1
    )

    if args.augment:
        print("\n2. Augmenting training data...")
        x_train, y_train = data_loader.augment_data(x_train, y_train, augmentation_factor=2)
        print(f"Augmented training set: {x_train.shape[0]} samples")

    print("\n3. Building lightweight emotion classifier...")
    classifier = EmotionClassifier(input_shape=(48, 48, 1), num_classes=7)
    model = classifier.build_lightweight_model()
    model.summary()
    print(f"Total parameters: {classifier.get_model_parameters():,}")

    print("\n4. Compiling model...")
    classifier.compile_model(learning_rate=args.learning_rate)

    print("\n5. Training model...")
    history = classifier.train(
        x_train, y_train,
        x_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\n6. Evaluating model on test set...")
    evaluator = ModelEvaluator(emotion_labels=classifier.emotion_labels)
    evaluation_results = evaluator.evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")

    print("\n7. Saving confusion matrix...")
    evaluator.plot_confusion_matrix(
        evaluation_results['confusion_matrix'],
        save_path='reports/confusion_matrix.png'
    )

    print("\n8. Exporting and quantizing models...")

    model.save('models/emotion_classifier.h5')

    tflite_path = 'models/emotion_classifier_int8.tflite'
    tflite_size = classifier.export_tflite(tflite_path, quantize=True)

    tflite_fp32_path = 'models/emotion_classifier_fp32.tflite'
    fp32_size = classifier.export_tflite(tflite_fp32_path, quantize=False)

    try:
        import tf2onnx
        onnx_path = 'models/emotion_classifier.onnx'
        onnx_size = classifier.export_onnx(onnx_path)
    except ImportError:
        print("tf2onnx not installed, skipping ONNX export")
        onnx_size = 0

    print("\n9. Benchmarking inference...")
    inference_engine = EmotionInference(tflite_path, model_type='tflite')

    test_image = np.random.rand(48, 48, 3).astype(np.float32)
    test_image = (test_image * 255).astype(np.uint8)

    benchmark_results = inference_engine.benchmark_inference(test_image, num_iterations=100)

    print(f"\nInference Benchmarks (INT8 Quantized):")
    print(f"  Mean latency: {benchmark_results['mean_latency_ms']:.2f} ms")
    print(f"  Std latency: {benchmark_results['std_latency_ms']:.2f} ms")
    print(f"  Median latency: {benchmark_results['median_latency_ms']:.2f} ms")

    print("\n10. Generating performance report...")
    evaluator.save_metrics_to_file(
        evaluation_results,
        benchmark_results,
        tflite_size,
        'reports/performance_metrics.txt'
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"✓ Top-1 Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"✓ Model Size (INT8): {tflite_size:.2f} MB")
    print(f"✓ Model Size (FP32): {fp32_size:.2f} MB")
    print(f"✓ Inference Latency: {benchmark_results['mean_latency_ms']:.2f} ± {benchmark_results['std_latency_ms']:.2f} ms")
    print("\nArtifacts saved:")
    print("  - models/emotion_classifier_int8.tflite")
    print("  - models/emotion_classifier_fp32.tflite")
    print("  - models/emotion_classifier.h5")
    if onnx_size > 0:
        print("  - models/emotion_classifier.onnx")
    print("  - reports/confusion_matrix.png")
    print("  - reports/performance_metrics.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train emotion classification model')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing FER2013 dataset')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')

    args = parser.parse_args()
    main(args)