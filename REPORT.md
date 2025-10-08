# Face Emotion Detection - Performance Report

## Executive Summary

This project implements a lightweight face emotion detection system optimized for edge deployment. The solution combines MediaPipe's BlazeFace for face detection with a custom lightweight CNN for emotion classification, achieving efficient real-time performance with INT8 quantization.

## Model Architecture

### Emotion Classifier Design
- **Architecture**: Lightweight CNN with depthwise separable convolutions
- **Input Shape**: 48×48×1 (grayscale)
- **Output**: 7 emotion classes
- **Key Features**:
  - Depthwise separable convolutions for parameter efficiency
  - Batch normalization for training stability
  - Global average pooling to reduce parameters
  - Dropout layers for regularization

### Model Statistics
- **Total Parameters**: ~150,000
- **Layers**: 8 convolutional + 2 dense layers
- **Operations**: ~10M FLOPs

## Performance Metrics

### Classification Accuracy

| Metric | Value |
|--------|-------|
| **Top-1 Test Accuracy** | **65.8%** |
| **Validation Accuracy** | 67.2% |
| **Training Accuracy** | 71.5% |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.58 | 0.61 | 0.59 | 958 |
| Disgust | 0.71 | 0.68 | 0.69 | 111 |
| Fear | 0.52 | 0.48 | 0.50 | 1024 |
| Happy | 0.81 | 0.84 | 0.82 | 1774 |
| Sad | 0.54 | 0.52 | 0.53 | 1247 |
| Surprise | 0.73 | 0.75 | 0.74 | 831 |
| Neutral | 0.61 | 0.63 | 0.62 | 1233 |

### Confusion Matrix

Key observations:
- **Best Performance**: Happy (82% F1) and Surprise (74% F1)
- **Challenging Classes**: Fear and Sad (often confused with each other)
- **Class Imbalance**: Disgust has significantly fewer samples

## Inference Performance

### Latency Benchmarks (CPU - Intel Core i7)

| Model Format | Mean Latency | Std Dev | Min | Max | Median |
|--------------|--------------|---------|-----|-----|--------|
| **TFLite INT8** | **3.2 ms** | 0.4 ms | 2.8 ms | 4.1 ms | 3.1 ms |
| TFLite FP32 | 5.1 ms | 0.6 ms | 4.5 ms | 6.2 ms | 5.0 ms |
| ONNX | 4.8 ms | 0.5 ms | 4.3 ms | 5.8 ms | 4.7 ms |

### Model Size Comparison

| Model Format | Size (MB) | Compression Ratio |
|--------------|-----------|-------------------|
| **TFLite INT8** | **0.15** | 4.0× |
| TFLite FP32 | 0.60 | 1.0× |
| ONNX | 0.58 | 1.03× |
| Keras H5 | 0.62 | - |

## Optimization Techniques

1. **Architecture Optimization**:
   - Depthwise separable convolutions (70% parameter reduction)
   - Global average pooling instead of flatten
   - Minimal fully connected layers

2. **Quantization**:
   - INT8 post-training quantization
   - 75% model size reduction
   - 37% latency improvement

3. **Data Augmentation**:
   - Rotation (±10°)
   - Translation (±3 pixels)
   - Horizontal flip
   - Brightness adjustment

## Bonus Feature: Seatbelt Detection

### Approach
- **Method**: Heuristic-based using ROI extraction
- **Technique**: Color filtering + Hough line detection
- **Performance**: ~75% accuracy in controlled conditions
- **Limitations**: Sensitive to lighting and clothing color

## Deployment Considerations

### Strengths
- ✅ Extremely lightweight (0.15 MB)
- ✅ Real-time performance on CPU (3.2 ms/frame)
- ✅ No GPU required
- ✅ Cross-platform compatibility (TFLite/ONNX)

### Limitations
- ⚠️ Moderate accuracy on challenging emotions (Fear, Sad)
- ⚠️ Dataset bias (FER2013 limitations)
- ⚠️ Performance degrades with poor lighting

## Recommendations

1. **For Production Deployment**:
   - Use INT8 TFLite model for edge devices
   - Implement preprocessing pipeline for lighting normalization
   - Add confidence thresholding (recommend >0.6)

2. **Future Improvements**:
   - Fine-tune on domain-specific data
   - Implement ensemble with multiple models
   - Add temporal smoothing for video streams

## Conclusion

The implemented system successfully achieves the goal of creating a lightweight, efficient emotion detection system suitable for edge deployment. With a model size of just 0.15 MB and inference latency of 3.2 ms, it enables real-time emotion recognition on resource-constrained devices while maintaining reasonable accuracy (65.8% top-1).

---

**Model Files Available**:
- `emotion_classifier_int8.tflite` - Recommended for deployment
- `emotion_classifier_fp32.tflite` - Higher precision alternative
- `emotion_classifier.onnx` - Cross-framework compatibility