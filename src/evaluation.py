import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple, Dict, List
import pandas as pd


class ModelEvaluator:
    def __init__(self, emotion_labels: List[str]):
        self.emotion_labels = emotion_labels

    def evaluate_model(self, model, x_test, y_test) -> Dict:
        y_pred_probs = model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                     target_names=self.emotion_labels,
                                     output_dict=True)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix - Emotion Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_training_history(self, history, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_metrics_report(self, evaluation_results: Dict,
                              benchmark_results: Dict,
                              model_size: float) -> str:
        report = []
        report.append("# Emotion Classification Model - Performance Report\n")
        report.append("## Model Metrics\n")
        report.append(f"**Top-1 Accuracy**: {evaluation_results['accuracy']:.4f}\n")
        report.append("\n## Per-Class Performance\n")

        df = pd.DataFrame(evaluation_results['classification_report']).transpose()
        report.append(df.to_string())

        report.append("\n\n## Inference Performance (CPU)\n")
        report.append(f"**Mean Latency**: {benchmark_results['mean_latency_ms']:.2f} ms\n")
        report.append(f"**Std Latency**: {benchmark_results['std_latency_ms']:.2f} ms\n")
        report.append(f"**Min Latency**: {benchmark_results['min_latency_ms']:.2f} ms\n")
        report.append(f"**Max Latency**: {benchmark_results['max_latency_ms']:.2f} ms\n")
        report.append(f"**Median Latency**: {benchmark_results['median_latency_ms']:.2f} ms\n")

        report.append(f"\n## Model Size\n")
        report.append(f"**Quantized Model Size**: {model_size:.2f} MB\n")

        return '\n'.join(report)

    def save_metrics_to_file(self, evaluation_results: Dict,
                            benchmark_results: Dict,
                            model_size: float,
                            output_path: str):
        report = self.generate_metrics_report(evaluation_results,
                                             benchmark_results,
                                             model_size)
        with open(output_path, 'w') as f:
            f.write(report)