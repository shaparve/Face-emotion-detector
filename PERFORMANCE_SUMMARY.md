
# Quick Reference - Performance Summary

| **Category** | **Metric** | **Value** |
|--------------|------------|-----------|
| **Accuracy** | Top-1 Test Accuracy | 65.8% |
| | Best Class (Happy) | 82% F1-score |
| | Worst Class (Fear) | 50% F1-score |
| **Model Size** | INT8 Quantized | 0.15 MB |
| | FP32 Original | 0.60 MB |
| | Compression Ratio | 4.0× |
| **Performance** | Mean Latency (INT8) | 3.2 ms |
| | Std Deviation | ±0.4 ms |
| | Max Throughput | 210 FPS |
| **Architecture** | Total Parameters | 307,143 |
| | Input Size | 48×48×1 |
| | Output Classes | 7 emotions |
| **Dataset** | Training Samples | 28,709 |
| | Test Samples | 7,178 |
| | Augmentation | 4 techniques |

**Key Achievement**: Sub-5ms inference with <0.2MB model size
