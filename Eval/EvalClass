import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

class Evaluation:
    def __init__(self, X, y, dataset, encoder, classifier='mlp', kfold=5):
        self.dataset = dataset
        self.encoder = encoder
        self.classifier = classifier
        self.kfold = kfold
        self.X = X
        self.y = y
        self.nclasses = len(np.unique(y))

        # Perform evaluation
        self.test_accuracy, self.testresults_acc = self.sentEval()

    def sentEval(self):
        if self.classifier == 'mlp':
            # Assuming InnerKFoldClassifier and its configuration is defined elsewhere
            from validation import InnerKFoldClassifier
            classifier_config = {
                'nhid': 0,
                'optim': 'rmsprop',
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 5
            }
            config = {
                'nclasses': self.nclasses,
                'seed': 2,
                'usepytorch': True,
                'classifier': classifier_config,
                'nhid': classifier_config['nhid'],
                'kfold': self.kfold,
                'drawcm': True
            }
            clf = InnerKFoldClassifier(self.X, self.y, config)
            dev_accuracy, test_accuracy, testresults_acc, cm_data = clf.run()
            

        elif self.classifier == 'lr':
            testresults_acc = []
            regs = [2**t for t in range(-2, 4, 1)]
            skf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=1111)
            innerskf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=1111)

            for train_idx, test_idx in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                scores = []
                for reg in regs:
                    regscores = []
                    for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                        X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                        y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                        clf = LogisticRegression(C=reg, random_state=0, max_iter=100000)
                        clf.fit(X_in_train, y_in_train)
                        regscores.append(clf.score(X_in_test, y_in_test))
                    scores.append(round(100 * np.mean(regscores), 5))

                optreg = regs[np.argmax(scores)]
                clf = LogisticRegression(C=optreg, random_state=0, max_iter=100000)
                clf.fit(X_train, y_train)
                f_acc = round(100 * clf.score(X_test, y_test), 2)
                testresults_acc.append(f_acc)
            test_accuracy = round(np.mean(testresults_acc), 2)
        else:
            raise ValueError("Unknown classifier type. Supported classifiers are 'mlp' and 'lr'.")

        return test_accuracy, testresults_acc


