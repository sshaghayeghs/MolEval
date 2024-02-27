from re import X
import os
import json
import numpy as np
import pandas as pd

# from whitening import whiten
from Eval.validation import InnerKFoldClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

class Evaluation:
    def __init__(self, X, y, eval_mlp=False):
            self.X = X
            self.y = y
            self.eval_mlp = eval_mlp
            self.classifier = 'mlp' if self.eval_mlp else 'lr'
            self.results = []
            self.kfold = 5
            self.nclasses = len(np.unique(self.y))
            
            acc, acc_list = self.sentEval(self.X, self.y, self.kfold, self.classifier, self.nclasses)
            result = {  'classifier': self.classifier,
                        'accuracy': acc,
                        'accuracy_list': acc_list,
                        'kfold': self.kfold
                    }
            self.results.append(result)


    def sentEval(self, X, y, kfold, classifier, nclasses):
        if(classifier == 'mlp'):
            classifier = {
                'nhid': 0,
                'optim': 'rmsprop',
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 5
            }
            config = {
                'nclasses': nclasses,
                'seed': 2,
                'usepytorch': True,
                'classifier': classifier,
                'nhid': classifier['nhid'],
                'kfold': kfold            }
            clf = InnerKFoldClassifier(X, y, config)
            dev_accuracy, test_accuracy, testresults_acc, cm_data = clf.run()
            
        elif(classifier == 'lr'):
            testresults_acc = []
            regs = [2**t for t in range(-2, 4, 1)]
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1111)
            innerskf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1111)

            for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                scores = []
                for reg in regs:
                    regscores = []
                    for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                        X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                        y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                        clf = LogisticRegression(C=reg, random_state=0, max_iter=100000)
                        clf.fit(X_in_train, y_in_train)
                        score = clf.score(X_in_test, y_in_test)
                        regscores.append(score)
                    # print(f'\t L2={reg} , fold {i} of {kfold}, score {score}')
                    scores.append(round(100*np.mean(regscores), 5))

                optreg = regs[np.argmax(scores)]
                # print('Best param found at split {0}:  L2 regularization = {1} with score {2}'.format(i, optreg, np.max(scores)))
                clf = LogisticRegression(C=optreg, random_state=0, max_iter=100000)
                clf.fit(X_train, y_train)

                f_acc = round(100*clf.score(X_test, y_test), 2)
                print(f'\taccuracy of {i} fold: {f_acc}')
                testresults_acc.append(f_acc)
            test_accuracy = round(np.mean(testresults_acc), 2)
        else:
            raise Exception("unknown classifier")

        return test_accuracy, testresults_acc
