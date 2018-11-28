from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

def grid_search(estimator, params, X, y):

    scores = []
    models = []
    for param_kernel in params:
        if param_kernel['kernel'] == 'linear':
            for C in param_kernel['C']:
                estimator.set_params(svc__C=C, svc__kernel='linear')
                score = cross_validation(estimator, X, y)
                scores.append(score)
                models.append(estimator)
                        
        if param_kernel['kernel'] == 'rbf':
            for C in param_kernel['C']:    
                for gamma in param_kernel['gamma']:
                    estimator.set_params(svc__C=C, svc__gamma=gamma, svc__kernel='linear')
                    score = cross_validation(estimator, X, y)
                    scores.append(score)
                    models.append(estimator)
    
    max_idx = np.array(scores).argmax()

    return scores[max_idx], models[max_idx]

        

def cross_validation(estimator, X, y):
    kfold = StratifiedKFold(n_splits=2, random_state=1).split(X,y)
    scores = []
    for k, (train, test) in enumerate(kfold):
        estimator.fit(X[train], y[train])
        score = accuracy_score(y[test], estimator.predict(X[test]))
        scores.append(score)
    
    return np.array(scores).mean()
        