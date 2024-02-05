import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import  f1_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
class Model:
    def __init__(self,threshold = 0.5, classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3)):
        # Define ensemble of models
        model_a = GaussianNB()
        model_b = LinearDiscriminantAnalysis()
        model_c = DecisionTreeClassifier(max_leaf_nodes=200)
        model_d = RandomForestClassifier(max_leaf_nodes=200)
        model_e = LogisticRegression(C=0.02, max_iter=1000)
        model_g = XGBClassifier(n_estimators=250, max_depth=2)
        model_h = AdaBoostClassifier(learning_rate=0.02, n_estimators=200)

        # Create a list of tuples with (name, model) for the VotingClassifier
        self.classifier = VotingClassifier(estimators=[("nb", model_a), ("lda", model_b), ("dt", model_c), ("rf", model_d), ("lr", model_e), ("xgb", model_g), ("adb", model_h)],
        voting="soft", weights=[4, 1, 1, 1, 3, 5, 4] )
        self.threshold = threshold

    def fit(self, X_train, y_train):
        # Fit the ensemble model
        self.classifier.fit(X_train, y_train)

        # Find optimal threshold using ROC curve
        probas = self.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, probas)
        optimal_idx = np.argmax(tpr + (1 - fpr))  
        self.threshold = thresholds[optimal_idx]
        # print(f"Threshold = {self.threshold}")

    def predict(self, X):
        # Predict using the ensemble model with the optimal threshold
        probas = self.classifier.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def score(self, X_test, y_test, show_metrics = False):
        # Evaluate model performance
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)

        if show_metrics:
            true_positive = sum((a == 1 and b == 1) for a, b in zip(y_test, y_pred))
            false_positive = sum((a == 0 and b == 1) for a, b in zip(y_test, y_pred))
            true_negative = sum((a == 0 and b == 0) for a, b in zip(y_test, y_pred))
            false_negative = sum((a == 1 and b == 0) for a, b in zip(y_test, y_pred))
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
                
            metrics = {"Sensitivity": [round(sensitivity, 3)],
                      "Specificity": [round(specificity, 3)],
                      "Accuracy": [round(self.classifier.score(X_test, y_test), 3)],
                      "F1-score": [round(f1_score(y_test, y_pred), 3)],

                      "True positive": [true_positive],
                      "False negative": [false_negative],
                      "True negative": [true_negative],
                      "False positive": [false_positive]}
            print(pd.DataFrame(metrics).T)

            return metrics
        else:
            return accuracy
    def predict_proba(self, X_test):
        # Predict probabilities using the ensemble model
        return self.classifier.predict_proba(X_test)



