from sklearn.neural_network import MLPClassifier
import numpy as np


class MLP():
    def __init__(self, *args, **kwargs):
        self.mlp = MLPClassifier(*args, **kwargs)

    def fit(self, x, y):
        return self.mlp.fit(x, y)

    def decision_function(self, x):
        return self.mlp.predict_proba(x)[:, 1] - 0.5


def nn_classify(x, y):
    #########################################
    ##          you code here              ##
    #########################################
    import pickle
    import os
    if os.path.isfile('nn.pkl'):
        print('-> Restore exist model from {}'.format('nn.pkl'))
        with open('nn.pkl', 'rb') as f:
            clf = pickle.load(f)
        return clf

    clf = MLP(hidden_layer_sizes=[256], alpha=0.992e-4,
              early_stopping=True, batch_size=1024, tol=1e-5)
    clf.fit(x, y)

    with open('clf_nn.pkl', 'wb') as f:
        pickle.dump(clf, f)
    #########################################
    ##          you code here              ##
    #########################################

    return clf