from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

class SVMTrainer:
    def __init__(self, cv_splits=5, cv_random_state=100, param_grid=None):
        """
        Initializes the SVM trainer with grid search parameters.
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            }
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.cv_random_state = cv_random_state
        self.best_params = None
        self.best_model = None

    def train(self, X_train, y_train):
        """
        Trains the SVM using GridSearchCV with cross-validation and returns
        the best estimator and hyperparameters.
        """
        svm_classifier = SVC()
        folds = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.cv_random_state)
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=self.param_grid, cv=folds)
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        print("Best Hyperparameters:", self.best_params)
        return self.best_model, self.best_params
