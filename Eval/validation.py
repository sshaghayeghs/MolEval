from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from Eval.classifier import MLP  # Ensure this is your actual MLP classifier

# Check sklearn version
assert sklearn.__version__ >= "0.18.0", "sklearn must be version >= 0.18.0"

def get_classif_name(classifier_config, usepytorch):
    """ Generate a name for the classifier based on its configuration. """
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config.get('nhid', 0)
        optim = classifier_config.get('optim', 'adam')
        bs = classifier_config.get('batch_size', 64)
        modelname = f'pytorch-MLP-nhid{nhid}-{optim}-bs{bs}'
    return modelname

class InnerKFoldClassifier(object):
    def __init__(self, X, y, config):
        """ Initialize the InnerKFoldClassifier with data and configuration. """
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config.get('seed', 0)
        self.devresults = []
        self.testresults = []
        self.dev_f1_scores = []
        self.test_f1_scores = []
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.k = config.get('kfold', 5)

    def run(self):
        """ Run the inner k-fold cross-validation process. """
        logging.info(f'Training {self.modelname} with (inner) {self.k}-fold cross-validation')

        regs = [10 ** t for t in range(-5, -1)] if self.usepytorch else [2 ** t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)

        for train_idx, test_idx in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            f1_scores = []

            for reg in regs:
                regscores = []
                reg_f1_scores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]

                    try:
                        if self.usepytorch:
                            clf = MLP(self.classifier_config, inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg, seed=self.seed)
                            clf.fit(X_in_train, y_in_train, validation_data=(X_in_test, y_in_test))
                        else:
                            clf = LogisticRegression(C=reg, random_state=self.seed)
                            clf.fit(X_in_train, y_in_train)
                    except Exception as e:
                        logging.error(f"Error during model training: {e}")
                        continue

                    y_pred_in_test = clf.predict(X_in_test)
                    regscores.append(clf.score(X_in_test, y_in_test))
                    reg_f1_scores.append(f1_score(y_in_test, y_pred_in_test, average='weighted'))

                scores.append(np.mean(regscores))
                f1_scores.append(np.mean(reg_f1_scores))

            optreg = regs[np.argmax(scores)]
            logging.info(f'Best param found at split: l2reg = {optreg} with score {np.max(scores)}')
            self.devresults.append(np.max(scores))
            self.dev_f1_scores.append(np.mean(f1_scores))

            try:
                if self.usepytorch:
                    clf = MLP(self.classifier_config, inputdim=self.featdim, nclasses=self.nclasses, l2reg=optreg, seed=self.seed)
                    clf.fit(X_train, y_train, validation_split=0.05)
                else:
                    clf = LogisticRegression(C=optreg, random_state=self.seed)
                    clf.fit(X_train, y_train)
            except Exception as e:
                logging.error(f"Error during model training: {e}")
                continue

            self.testresults.append(clf.score(X_test, y_test))
            y_pred_test = clf.predict(X_test)
            self.test_f1_scores.append(f1_score(y_test, y_pred_test, average='weighted'))

        return np.mean(self.devresults), np.mean(self.testresults), np.mean(self.dev_f1_scores), np.mean(self.test_f1_scores)


