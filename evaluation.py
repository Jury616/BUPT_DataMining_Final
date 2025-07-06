# evaluation.py
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_test, y_pred, target_names=None, model_name="Model", feature_type="Features"):
    """
    评估分类器性能并返回指标字典
    :param y_test: 真实标签
    :param y_pred: 预测标签
    :param target_names: 类别名称列表 (可选)
    :param model_name: 模型名称 (用于打印)
    :param feature_type: 特征类型 (用于打印)
    :return: 包含评估指标的字典
    """
    accuracy = accuracy_score(y_test, y_pred)
    # Set zero_division=0 to avoid warnings when a class has no predicted samples for precision/recall.
    precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    precision_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    print(f"--- Performance: {model_name} on {feature_type} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision_w:.4f}")
    print(f"Recall (Weighted): {recall_w:.4f}")
    print(f"F1 Score (Weighted): {f1_w:.4f}")
    print(f"Precision (Macro): {precision_m:.4f}")
    print(f"Recall (Macro): {recall_m:.4f}")
    print(f"F1 Score (Macro): {f1_m:.4f}")
    print("\nClassification Report:\n", report_str)

    metrics = {
        "Model": model_name,
        "Features": feature_type,
        "Accuracy": accuracy,
        "Precision (Weighted)": precision_w,
        "Recall (Weighted)": recall_w,
        "F1 Score (Weighted)": f1_w,
        "Precision (Macro)": precision_m,
        "Recall (Macro)": recall_m,
        "F1 Score (Macro)": f1_m,
        "Classification Report String": report_str
    }
    return metrics