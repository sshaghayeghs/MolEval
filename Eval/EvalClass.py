import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Assuming InnerKFoldClassifier and its configuration is defined elsewhere
from Eval.validation import InnerKFoldClassifier

class Evaluation:
    def __init__(self, X, y, classifier='mlp', kfold=5, random_state=1111):
        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        
        self.classifier = classifier
        self.kfold = kfold
        self.X = X
        self.y = y
        self.nclasses = len(np.unique(y))
        self.random_state = random_state

        # Perform evaluation
        self.test_accuracy, self.test_f1, self.testresults_acc, self.testresults_f1 = self.sentEval()

    def sentEval(self):
        if self.classifier == 'mlp':
            # Classifier configuration for 'mlp'
            classifier_config = {
                'nhid': 0,
                'optim': 'rmsprop',
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 5,
                'random_state': self.random_state  # Ensuring consistency in random state
            }
            config = {
                'nclasses': self.nclasses,
                'seed': self.random_state,  # Using consistent random state
                'usepytorch': True,
                'classifier': classifier_config,
                'kfold': self.kfold
            }
            clf = InnerKFoldClassifier(self.X, self.y, config)
            dev_accuracy, test_accuracy, testresults_acc, dev_f1, test_f1 = clf.run()

        return dev_accuracy, test_accuracy, testresults_acc, dev_f1, test_f1
