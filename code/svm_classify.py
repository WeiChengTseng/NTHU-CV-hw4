from sklearn import svm
import numpy as np

def svm_classify(x, y, reload=True):
    '''
    FUNC: train SVM classifier with input data x and label y
    ARG:
        - x: input data, HOG features
        - y: label of x, face or non-face
    RET:
        - clf: a SVM classifier using sklearn.svm. (You can use your favorite
               SVM library but there will be some places to be modified in
               later-on prediction code)
    '''
    #########################################
    ##          you code here              ##
    #########################################
    import pickle as pkl
    import os
    import sklearn

    reload = False
    if os.path.isfile('SVM_linear.pkl') and reload:
        print('-> Restore exist model from {}'.format('SVM_linear.pkl'))
        with open('SVM_linear.pkl', 'rb') as f:
            clf = pkl.load(f)
        return clf

    print('-> Establish a new linear SVM model.')
    parameters_set = {'C': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.7, 2.0, 2.5, 10]}
    SVM_linear = svm.LinearSVC(tol=1e-5, random_state=0)
    gv = sklearn.model_selection.GridSearchCV(SVM_linear, 
                                               parameters_set, cv=8, 
                                               scoring='accuracy',
                                               n_jobs=-1)
    print('-> Start running cross validation.')
    gv.fit(x, y[:, 0])
    print('-> Best paramters searched by GridSearchCV:', gv.best_params_)
    clf = svm.LinearSVC(tol=1e-5, random_state=0, C=gv.best_params_['C'])
    clf.fit(x, y[:, 0])

    with open('SVM_linear.pkl', 'wb') as model:
        print('-> Save model to {}'.format('SVM_linear.pkl'))
        pkl.dump(clf, model)
    #########################################
    ##          you code here              ##
    #########################################

    return clf
