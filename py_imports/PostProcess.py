# -*- coding: utf-8 -*-
import numpy as np
from   sklearn  import metrics
#%%
class PostProcess:
    
    def __init__(self, yPred, yPred_proba, yTrue, fileOut):
        """
        Function used for calculating and saving the binary classification metrics in a file.\n
        Keyword arguments:\n
            yPred       : Predicted labels.\n
            yPred_proba : Predicted probabilities per class.\n
            yTrue       : True labels.\n
            fileOut     : Full filename path where the metrics are to be saved.\n
        """
        self.yPred = yPred
        self.yPred_proba =  yPred_proba
        self.yTrue = yTrue
        self.fileOut= fileOut
        
    def calculate_top_5_accuracy(self):
        """
        Function used for calculating top 5 accuracy.\n
        """
        acc_top_5 = 0
        top5_predictions = np.argsort(self.yPred_proba, axis=1)[:, -5:]
        for i in range(len(self.yPred)):
            if self.yTrue[i] in top5_predictions[i]:
                acc_top_5 += 1
        acc_top_5 /= len(self.yPred)
        
        return acc_top_5

    def calculate_multiclass_metrics(self):
        """
        Function used for calculating and saving the classification metrics and save a file.\n
        """
        accuracy  = metrics.accuracy_score(self.yTrue, self.yPred)
        acc_top5  = self.calculate_top_5_accuracy()
        precision = metrics.precision_score(self.yTrue, self.yPred, average="weighted")
        recall    = metrics.recall_score(self.yTrue, self.yPred, average="weighted")
        f1        = metrics.f1_score(self.yTrue, self.yPred, average="weighted")
        Metrics   = "---- Multiclass classification metrics ----\n"
        Metrics  += f"{'Accuracy':<20s} : {accuracy}\n"
        Metrics  += f"{'Top 5 Accuracy':<20s} : {acc_top5}\n"
        Metrics  += f"{'Precision:':<20s} : {precision}\n"
        Metrics  += f"{'Recall':<20s} : {recall}\n"
        Metrics  += f"{'F1 Score':<20s} : {f1}\n"
        Metrics  += self.calculate_baseline_accuracy()
        
        with open(self.fileOut, "w") as fOut:
            fOut.write(Metrics)
       
        return Metrics
    
    def calculate_baseline_accuracy(self):
        """
        Function used for calculating the baseline accuracy of the given dataset.\n
        Keyword arguments:\n
            binary_labels :
        """
        unique_classes, class_counts = np.unique(self.yTrue, return_counts=True)
        majority_class_count = np.max(class_counts)
        total_instances = len(self.yTrue)
        baseline_accuracy = majority_class_count / total_instances
        
        baseline = "------ Baseline accuracy ------\n"
        baseline += f"{'Baseline accuracy':<20s} : {baseline_accuracy}\n"
        
        return baseline
