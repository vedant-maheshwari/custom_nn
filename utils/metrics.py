from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def binary_classification_metrics(y_true, y_pred):
    """
    y_true, y_pred should be 1D arrays with values 0 or 1.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    print("Binary Classification Metrics")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
