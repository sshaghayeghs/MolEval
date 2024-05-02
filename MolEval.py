import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc,mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings("ignore")
def load_emb(dataset, language_model, dataset_path=None, embedding_path=None):
    df = pd.read_csv(f"{dataset_path}{dataset}.csv")
    features_df = pd.read_csv(f"{embedding_path}{dataset}_{language_model}.csv")
    if 'Unnamed: 0' in features_df.columns:
        features_df = features_df.drop(columns=['Unnamed: 0'])
    targets = df.drop(columns=['SMILES']).to_numpy()
    features = features_df.to_numpy()
    ids = df['SMILES'].tolist()
    return features, targets



def evaluate_classification(features, targets, n_splits, task):
    # K-fold cross-validation setup
    if task == 'MultitaskClassification':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores = []
    aucrocs = []
    for train_index, test_index in kf.split(features, targets if task != 'MultitaskClassification' else None):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        if task == 'Classification':
            model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train.ravel())  # Assuming y_train is not multilabel
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
        elif task == 'MultitaskClassification':
            model = MultiOutputClassifier(LogisticRegression(max_iter=500))
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            try:
                y_test_proba = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T
            except NotFittedError:
                print("Model not fitted or single output model predict_proba called")
                continue

        [f1, aucroc] = get_results(y_test, y_test_pred, y_test_proba, task)
        f1_scores.append(f1)
        aucrocs.append(aucroc)


    return [np.mean(f1_scores),np.std(f1_scores),np.mean(aucrocs),np.std(aucrocs)]

def get_results(test_labels, y_pred, y_pred_proba, task):
    if task == 'MultitaskClassification':
        # Calculate metrics for multilabel data
        prec = precision_score(test_labels, y_pred, average='macro')
        recall = recall_score(test_labels, y_pred, average='macro')
        f1 = f1_score(test_labels, y_pred, average='macro')

        aucrocs = []
        for i in range(test_labels.shape[1]):
            if len(np.unique(test_labels[:, i])) > 1:  # only if there are both classes present
                aucrocs.append(roc_auc_score(test_labels[:, i], y_pred_proba[:, i]))
        aucroc = np.mean(aucrocs)

        precisions, recalls, _ = precision_recall_curve(test_labels.ravel(), y_pred_proba.ravel())
        aupr = auc(recalls, precisions)
    else:
        # Binary classification metrics
        prec = precision_score(test_labels, y_pred)
        recall = recall_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred)
        aucroc = roc_auc_score(test_labels, y_pred_proba)
        p, r, _ = precision_recall_curve(test_labels, y_pred_proba)
        aupr = auc(r, p)

    return [f1, aucroc]
def evaluate_regression(features, targets, n_splits=5):
    # Set up KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        # Model training
        model = Ridge()
        model.fit(X_train, y_train.ravel())  # Ensure y_train is properly shaped

        # Prediction and performance evaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store the scores
        r2_scores.append(r2)
        rmse_scores.append(rmse)

    return [np.mean(rmse_scores),np.std(rmse_scores), np.mean(r2_scores),np.std(rmse_scores)]


