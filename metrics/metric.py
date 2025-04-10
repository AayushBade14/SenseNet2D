from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

class ModelMetricsEvaluator:
    def __init__(self, model_name="Model"):
        self.model_name = model_name

    def evaluate(self, y_true, y_pred, display_plot=True):
        print(f"🔍 Evaluation Results for {self.model_name}:\n")
        
        # 1. Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # 2. Classification Report
        report = classification_report(y_true, y_pred)
        
        # 3. Confusion Matrix
        matrix = confusion_matrix(y_true, y_pred)

        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("\n Classification Report:")
        print(report)
        print("\n Confusion Matrix:")
        print(matrix)

        # 4. Plot Confusion Matrix
        if display_plot:
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {self.model_name}")
            plt.grid(False)
            plt.show()

        # Optional: return values for logging or further analysis
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": matrix
        }
