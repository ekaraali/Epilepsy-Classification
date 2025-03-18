import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, auc, roc_curve

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set, printing accuracy and the classification report.
    Returns accuracy, report, and the predictions.
    """
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    return accuracy, report, y_pred

def plot_confusion_matrix(y_test, y_pred):
    """
    Plots a confusion matrix with labels 3 and 4.
    """
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=[3, 4], columns=[3, 4])
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d",
                cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title('Model Accuracy: {}%'.format(np.round(accuracy_score(y_test, y_pred) * 100, 2)))
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred):
    """
    Plots the precision-recall curve and displays the AUC for precision-recall.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (AUC-PR={auc_pr:.2f})')
    plt.show()

def plot_roc_curve(y_test, y_pred):
    """
    Plots the ROC curve and calculates the AUC for ROC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
