from sklearn.feature_selection import SelectKBest, mutual_info_classif

class FeatureSelector:
    def __init__(self, k_features=50):
        """
        Initializes the feature selector with the desired number of features.
        """
        self.k_features = k_features
        self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k_features)

    def select_features(self, X_train, X_test, y_train, y_test):
        """
        Applies SelectKBest to choose the top k features on the provided
        training and test sets.
        """
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)
        print(f"Selected top {self.k_features} features.")
        return X_train_selected, X_test_selected, y_train, y_test
