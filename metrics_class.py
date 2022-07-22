import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import seaborn as sns


class Metrics(object):
    '''
    Class containing functions to plot the different metric curves (Precision-Recall, ROC AUC etc...)
    '''

    def __init__(self):
        pass

    @classmethod
    def roc_auc_curve(self, model, x, y, labels, gb=False):
        '''
        Function to plot the ROC AUC curves for binary or multiclass classification.
        Correct for standard machine learning models and Neural Networks.
        '''
        global lr_auc
        ns_probs = [0 for _ in range(len(y.reshape(-1, 1)))]
        if gb:
            lr_probs = model.predict_proba(x)
        else:
            lr_probs = model.predict(x)

        plt.figure(figsize=(10, 8))
        if len(labels) == 2:
            if gb:

                lr_auc = roc_auc_score(y, lr_probs[:, 1], average="weighted")
            else:
                lr_auc = roc_auc_score(y, lr_probs, average="weighted")
            ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            if gb:
                lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs[:, 1])
            else:
                lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)

            plt.plot(lr_fpr, lr_tpr, label=f'Class (area {round(lr_auc, 3)})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.legend()

        print('\nROC AUC=%.3f \n' % lr_auc)
        plt.show()

    @classmethod
    def confusion_matrix(self, model, y, x, labels):
        '''
        Compute the confusion matrix for binary or multiclass classification.
        Correct for standard machine learning models and Neural Networks.
        '''
        if len(labels) == 2:  # binary confusion matrix
            confu_matrix = pd.DataFrame(confusion_matrix(y, (model.predict(x) > 0.5).astype(int)), \
                                        columns=['Predicted Negative', "Predicted Positive"],
                                        index=['Actual Negative', 'Actual Positive'])
            print(confu_matrix)
            return confu_matrix

    @classmethod
    def precision_recall_curve(self, model, x, y, labels, gb=False):
        '''
        Function to plot the recall precision curves for binary or multiclass classification.
        Correct for standard machine learning models and Neural Networks.
        '''
        if gb:
            lr_probs = model.predict_proba(x)
        else:
            lr_probs = model.predict(x)

        print("\n")
        plt.figure(figsize=(10, 8))

        if len(labels) == 2:
            if gb:
                precision, recall, thresholds = precision_recall_curve(y, lr_probs[:,
                                                                          1])
                lr_f1 = f1_score(y, (lr_probs[:, 1] > 0.5).astype(int))
            else:
                precision, recall, thresholds = precision_recall_curve(y, lr_probs)
                lr_f1 = f1_score(y, (lr_probs > 0.5).astype(int))

            lr_auc = auc(recall, precision)
            print('Model: f1-score=%.3f AUC=%.3f' % (lr_f1, lr_auc))  # print f1-score and auc
            plt.plot(recall, precision, marker='.', label='Model')
            no_skill = len(y[y == 1]) / len(y)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        else:
            dummy_y = np_utils.to_categorical(y)
            dummy_lr = np_utils.to_categorical(lr_probs.argmax(-1))
            for i in enumerate(labels):
                precision, recall, thresholds = precision_recall_curve(dummy_y[:, i[0]], lr_probs[:, i[0]])
                lr_f1 = f1_score(dummy_y[:, i[0]], dummy_lr[:, i[0]])
                lr_auc = auc(recall, precision)
                print('Model class: %s --> f1-score=%.3f AUC=%.3f' % (i[1], lr_f1, lr_auc))
                plt.plot(recall, precision, label='Class %s' % (i[1]))
            no_skill = len(y[y >= 1]) / len(y)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        print("\n")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def plot_eval_xgb(self, model, labels):
        '''
        Function to plot the evaluation curves for xgboost models
        '''
        results = model.evals_result()
        if len(labels) > 2:
            log_ = "mlogloss"
            error_ = "merror"
        else:
            log_ = "logloss"
            error_ = "error"

        epochs = len(results['validation_0'][error_])
        x_axis = range(0, epochs)

        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.plot(x_axis, results['validation_0'][log_], label='Train')
        plt.plot(x_axis, results['validation_1'][log_], label='Test')
        plt.ylabel('Log Loss')
        plt.xlabel('Epochs')
        plt.title('XGBoost Log Loss')
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.subplot(222)
        plt.plot(x_axis, results['validation_0'][error_], label='Train')
        plt.plot(x_axis, results['validation_1'][error_], label='Test')
        plt.legend()
        plt.ylabel('Classification Error')
        plt.xlabel('Epochs')
        plt.title('XGBoost Classification Error')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    @classmethod
    def plot_confusion_matrix(self, cm, classes, normalized=True, cmap='bone'):
        '''
        Function to generate an heatmap of the confusion matrix
        '''
        plt.figure(figsize=[10, 8])
        norm_cm = cm
        if normalized:
            norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
            # plt.savefig('confusion-matrix.png')

    @classmethod
    def plot_history(self, history):
        '''
        Function to plot the learning curves of a neural network
        @param history: metrics of a neural network
        '''
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid(True)

        plt.subplot(222)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid(True)
        plt.show()

    @classmethod
    def metrics_deep_learning(self, model, history, x, y, labels):
        '''
        Function to plot the different metrics for the deep learning algorithms.
        '''
        self.plot_history(history)
        if len(labels) == 2:

            print(classification_report(y, (model.predict(x) > 0.5).astype(int), target_names=labels))
            print(
                f"\nThe balanced accuracy is : {round(100 * balanced_accuracy_score(y, (model.predict(x) > 0.5).astype(int)), 2)}%\n")
            print(
                f"\nThe Zero-one Loss is : {round(100 * zero_one_loss(y, (model.predict(x) > 0.5).astype(int)), 2)}%\n")
            print(
                f"\nExplained variance score: {round(explained_variance_score(y, (model.predict(x) > 0.5).astype(int)), 3)}\n")
            self.roc_auc_curve(model, x, y, labels)
            self.precision_recall_curve(model, x, y, labels)

            print(f"\nCohen's kappa: {round(100 * cohen_kappa_score(y, (model.predict(x) > 0.5).astype(int)), 2)}% \n")
            cm = confusion_matrix(y, (model.predict(x) > 0.5).astype(int))

            print("\nConfusion Matrix\n")
            self.plot_confusion_matrix(cm, labels)
        else:

            print(
                f"\nThe balanced accuracy is : {round(100 * balanced_accuracy_score(y, model.predict(x).argmax(axis=-1)), 2)}%\n")
            print(f"\nThe Zero-one Loss is : {round(100 * zero_one_loss(y, model.predict(x).argmax(axis=-1)), 2)}%\n")
            print(
                f"\nExplained variance score: {round(explained_variance_score(y, model.predict(x).argmax(axis=-1)), 3)}\n")
            self.roc_auc_curve(model, x, y, labels)
            self.precision_recall_curve(model, x, y, labels)

            print(f"\nCohen's kappa: {round(100 * cohen_kappa_score(y, model.predict(x).argmax(axis=-1)), 2)}%\n")
            cm = confusion_matrix(y, model.predict(x).argmax(axis=-1))

            print(classification_report(y, model.predict(x).argmax(axis=-1), target_names=labels))

            print("\nConfusion Matrix\n")
            self.plot_confusion_matrix(cm, labels)
