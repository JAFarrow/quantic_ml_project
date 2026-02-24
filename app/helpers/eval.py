from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def build_evaluation(y_true_raw, predictions):
    y_true = [int(v) for v in y_true_raw]
    y_pred = [int(p["prediction"]) for p in predictions]
    y_prob = [float(p["probability"]) for p in predictions]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)

    evaluation = {
        "available": True,
        "accuracy": float(acc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

    return evaluation
