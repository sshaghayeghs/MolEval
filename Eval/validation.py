from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
from classifier import MLP

import sklearn
assert (sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname


# Pytorch version
class InnerKFoldClassifier(object):
    """
    (train) split classifier : InnerKfold.
    """

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testresults = []
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'
                     .format(self.modelname, self.k))

        regs = [10 ** t for t in range(-5, -1)] if self.usepytorch else \
            [2 ** t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    if self.usepytorch:
                        clf = MLP(self.classifier_config, inputdim=self.featdim,
                                  nclasses=self.nclasses, l2reg=reg,
                                  seed=self.seed)
                        clf.fit(X_in_train, y_in_train,
                                validation_data=(X_in_test, y_in_test))
                    else:
                        clf = LogisticRegression(C=reg, random_state=self.seed)
                        clf.fit(X_in_train, y_in_train)
                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(round(100 * np.mean(regscores), 2))
            optreg = regs[np.argmax(scores)]
            logging.info('Best param found at split {0}: l2reg = {1} \
                with score {2}'.format(count, optreg, np.max(scores)))
            self.devresults.append(np.max(scores))

            if self.usepytorch:
                clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=self.nclasses, l2reg=optreg,
                          seed=self.seed)
                clf.fit(X_train, y_train, validation_split=0.05)
            else:
                clf = LogisticRegression(C=optreg, random_state=self.seed)
                clf.fit(X_train, y_train)

            
            self.testresults.append(round(100 * clf.score(X_test, y_test), 2))


        devaccuracy = round(np.mean(self.devresults), 2)
        testaccuracy = round(np.mean(self.testresults), 2)
        return devaccuracy, testaccuracy, self.testresults
