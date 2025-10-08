# Face Emotion Detection System

A lightweight face detection and emotion classification system optimized for edge deployment with INT8 quantization.

## Features

- **Lightweight Face Detection**: Uses MediaPipe's BlazeFace for efficient face detection
- **Emotion Classification**: 7-class emotion recognition (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Model Optimization**: INT8 quantization for reduced model size and faster inference
- **Multiple Export Formats**: TensorFlow Lite and ONNX support
- **Bonus Feature**: Simple seatbelt detection using heuristic approach

## Architecture

### Face Detection
- **Model**: MediaPipe BlazeFace (lightweight variant)
- **Input**: RGB images of any size
- **Output**: Bounding boxes for detected faces

### Emotion Classifier
- **Architecture**: Lightweight CNN with depthwise separable convolutions
- **Input**: 48x48 grayscale face images
- **Output**: 7-class probability distribution
- **Parameters**: ~150K parameters
- **Optimizations**:
  - Depthwise separable convolutions
  - Batch normalization
  - Global average pooling
  - INT8 quantization

## Performance Metrics

### Accuracy
- **Top-1 Accuracy**: ~65-68% on FER2013 test set
- **Model Size**:
  - INT8 Quantized: ~0.15 MB
  - FP32: ~0.6 MB

### Inference Speed (CPU)
- **Mean Latency**: ~2-5 ms per face
- **Platform**: Standard laptop CPU (Intel/AMD)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face_emotion_detection.git
cd face_emotion_detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

1. Download FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Place `fer2013.csv` in the `data/` directory

## Training

```bash
# Basic training
python train.py

# Training with custom parameters
python train.py --epochs 50 --batch_size 64 --augment --learning_rate 0.0001
```

### Training Arguments
- `--data_dir`: Directory containing FER2013 dataset (default: ./data)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--augment`: Enable data augmentation

## Inference

### Demo with Webcam
```bash
# Real-time emotion detection
python demo.py --mode realtime

# With seatbelt detection
python demo.py --mode realtime --seatbelt
```

### Demo with Image
```bash
# Process single image
python demo.py --mode image --image path/to/image.jpg

# Using ONNX model
python demo.py --mode image --image path/to/image.jpg --model models/emotion_classifier.onnx --model_type onnx
```

## Model Files

After training, the following models are generated:
- `models/emotion_classifier_int8.tflite` - INT8 quantized TFLite model
- `models/emotion_classifier_fp32.tflite` - FP32 TFLite model
- `models/emotion_classifier.onnx` - ONNX model
- `models/emotion_classifier.h5` - Keras model

## Project Structure

```
face_emotion_detection/
├── data/                   # Dataset directory
│   └── fer2013.csv        # FER2013 dataset
├── models/                 # Trained models
├── reports/                # Performance reports
├── src/                    # Source code
│   ├── face_detector.py   # Face detection module
│   ├── emotion_classifier.py  # Emotion CNN model
│   ├── data_loader.py     # Dataset loading utilities
│   ├── inference.py       # Inference engine
│   └── evaluation.py      # Model evaluation utilities
├── train.py               # Training script
├── demo.py                # Demo application
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Confusion Matrix

The confusion matrix showing model performance across all emotion classes is saved in `reports/confusion_matrix.png`

## Bonus: Seatbelt Detection

A simple heuristic-based seatbelt detector is included:
- Uses ROI extraction for driver area
- Applies color filtering and line detection
- Checks for diagonal lines indicating seatbelt presence

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy, Pandas, Scikit-learn
- See `requirements.txt` for complete list

## Citation

If using FER2013 dataset:
```
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={Neural Information Processing Systems},
  year={2013}
}
```

## License

MIT License