from preprocessing import EEGPreprocessor
from feature_selection import FeatureSelector
from svm_trainer import SVMTrainer
from evaluation import evaluate_model, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve

def main():
    # Update the path to your EEG dataset
    csv_path = "/content/EEG-data.csv"

    # Data preprocessing with stratified train-test split, outlier elimination, normalization, and SMOTE on the training set
    preprocessor = EEGPreprocessor(csv_path=csv_path)
    X_train, y_train, X_test, y_test = preprocessor.load_and_preprocess()

    # Feature selection
    selector = FeatureSelector(k_features=50)
    X_train_selected, X_test_selected, y_train, y_test = selector.select_features(X_train, X_test, y_train, y_test)

    # Train the SVM model with grid search
    trainer = SVMTrainer()
    best_model, best_params = trainer.train(X_train_selected, y_train)

    # Evaluate the model
    accuracy, report, y_pred = evaluate_model(best_model, X_test_selected, y_test)

    # Plot evaluation metrics
    plot_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)


if __name__ == "__main__":
    main()
