import argparse
import json
import os

import numpy as np
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


def get_data(train_val_test, data_path, split):
    patient_info = train_val_test[split]

    X = []
    y = []
    for patient in patient_info:
        cur_X = []
        for filename in patient_info[patient]['filenames']:
            cur_X.append(np.load(os.path.join(data_path, filename + '.npy')))
        X.extend(cur_X)

        target = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        for disease in patient_info[patient]['diseases']:
            target[int(train_val_test['class_mapping'][disease])] = 1
        y.extend([target] * len(cur_X))

    return np.stack(X), np.stack(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--use_scaler', default=False, action='store_true')
    parser.add_argument('--split_json',
                        type=str,
                        default='/fs/ess/PCON0023/eye3d/data/ukbiobank/train_val_test2.json',
                        help='path to train_val_test.json')
    parser.add_argument('--data_path', type=str, default='', help='data')
    args = parser.parse_args()

    with open(args.split_json, 'r') as f:
        train_val_test = json.load(f)

    X_train, y_train = get_data(train_val_test, args.data_path, 'train')
    X_val, y_val = get_data(train_val_test, args.data_path, 'val')
    X_test, y_test = get_data(train_val_test, args.data_path, 'test')

    best_model, best_penalty, results, test_roc_aucs = multilabel_logistic_regression_tuning(
        args, X_train, y_train, X_val, y_val, X_test, y_test,
        penalties=[0.01, 0.1, 1.0, 10.0, 100.0]
    )