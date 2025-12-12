import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
import warnings

warnings.filterwarnings('ignore')


def multilabel_logistic_regression_tuning(args, X_train, y_train, X_val, y_val, X_test, y_test, penalties):
    if args.use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    results = {}
    best_roc_auc = -1
    best_pr_auc = -1
    best_model = None
    best_penalty = None

    for penalty in penalties:
        model = OneVsRestClassifier(
            LogisticRegression(
                C=1 / penalty,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_val_pred_proba = model.predict_proba(X_val)

        roc_auc_scores = []
        pr_auc_scores = []
        for label_idx in range(y_val.shape[1]):
            score = roc_auc_score(y_val[:, label_idx], y_val_pred_proba[:, label_idx])
            roc_auc_scores.append(score)
            score = average_precision_score(y_val[:, label_idx], y_val_pred_proba[:, label_idx])
            pr_auc_scores.append(score)

        avg_roc_auc = np.mean(roc_auc_scores)
        avg_pr_auc = np.mean(pr_auc_scores)

        results[penalty] = {
            'avg_roc_auc': avg_roc_auc,
            'individual_scores': roc_auc_scores,
            'model': model
        }

        print(f"penalty: {penalty:8.2f} | C: {1 / penalty:8.4f} | Average ROC AUC: {avg_roc_auc:.4f}"
              f" | Average PR AUC: {avg_pr_auc:.4f}")

        if avg_roc_auc > best_roc_auc:
            best_roc_auc = avg_roc_auc
            best_pr_auc = avg_pr_auc
            best_penalty = penalty
            best_model = model
        elif avg_roc_auc == best_roc_auc:
            best_pr_auc = avg_pr_auc
            best_penalty = penalty
            best_model = model

    y_test_pred_proba = best_model.predict_proba(X_test)
    test_roc_aucs = []
    test_pr_aucs = []
    for label_idx in range(y_test.shape[1]):
        score = roc_auc_score(y_test[:, label_idx], y_test_pred_proba[:, label_idx])
        test_roc_aucs.append(score)
        score = average_precision_score(y_test[:, label_idx], y_test_pred_proba[:, label_idx])
        test_pr_aucs.append(score)

    test_avg_roc_auc = np.mean(test_roc_aucs)
    test_avg_pr_auc = np.mean(test_pr_aucs)

    print("=" * 60)
    print(f"\nbest_penalty: {best_penalty}")
    print(f"Best Average ROC AUC: {best_roc_auc:.4f}")
    print(f"Best Average ROC AUC: {best_pr_auc:.4f}")
    print(f"Test Best Average ROC AUC: {test_avg_roc_auc:.4f}")
    print(f"Test Best Average PR AUC: {test_avg_pr_auc:.4f}")

    return best_model, best_penalty, results, test_roc_aucs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--use_scaler', default=False, action='store_true')
    args = parser.parse_args()

    from sklearn.datasets import make_multilabel_classification

    X, y = make_multilabel_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_labels=8,
        random_state=42
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    best_model, best_penalty, results, test_roc_aucs = multilabel_logistic_regression_tuning(
        args, X_train, y_train, X_val, y_val, X_test, y_test,
        penalties=[0.01, 0.1, 1.0, 10.0, 100.0]
    )