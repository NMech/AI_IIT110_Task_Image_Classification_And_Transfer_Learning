# -*- coding: utf-8 -*-
from   BasicFigureTemplate import BasicFigure_Template
from   sklearn             import metrics
import seaborn           as sn
import matplotlib.pyplot as plt
import numpy             as np
#%%
class ClassificationMetricsPlot(BasicFigure_Template):
    
    def __init__(self, yTrue, FigureProperties=["a3paper","pdf","landscape","white",0.5],
                 FontSizes=[20.0,16.0,14.0,10.0]):
        """
        Initialization.\n
        Keyword arguments:\n
            yTrue : True labels.\n
        """
        self.yTrue   = yTrue
        self.nLabels = len(np.unique(yTrue))
        BasicFigure_Template.__init__(self,FigureProperties,FontSizes)
        self.dim1, self.dim2  = self.FigureSize() 
        
    def __metrics(self, yPred):
        """
        Auxiliary function used for calculating different metrics.
        """
        if self.nLabels != 2:
            aveRage = "weighted" 
        else:
            aveRage = "binary"
        accuracy = round(metrics.accuracy_score(self.yTrue, yPred), 3)
        precision= round(metrics.precision_score(self.yTrue, yPred, average=aveRage), 3)
        recall   = round(metrics.recall_score(self.yTrue, yPred, average=aveRage), 3)
        f1       = round(metrics.f1_score(self.yTrue, yPred, average=aveRage), 3)
        
        return accuracy, precision, recall, f1
    
    def Confusion_Matrix_Plot(self, yPred, CMat, normalize=False, labels="auto", cMap="default", Title="",
                        Rotations=[0.,0.], annotation=True, savePlot=["False","<filepath>","<filename>"]):
        """
        Implementation of confusion matrix using seaborn's heatmap. See also metrics.ConfusionMatrixDisplay.\n
        Keyword arguments:\n
            yPred     : Predicted labels from classifier.\n
            CMat      : Confusion matrix calculated from sklearn.metrics.confusion_matrix.\n
            normalize : Boolean. If True then the values of the confusion matrix are normalized.\n
            labels    : Labels of the classes. By default "auto".\n
            cMap      : Cmap to be used.\n
            Title     : Title used in the plot.\n
            Rotations : x,y-ticks rotations. Default values 0. and 0. degrees.\n
            annotation: In cell annotation.\n
            savePlot  : list conatining the following.\n
                        * Save plot boolean.\n
                        * Filepath where the diagram will be saved.\n
                        * Filename (without the filetype) of the diagram to be plotted.\n
        Returns fig, ax.
        """
        if normalize == True:
            CMat = CMat/sum(sum(CMat))*100#self.nLabels*100.
        if cMap == "default":
            cMap = plt.cm.Blues
 
        fig,ax = plt.subplots(figsize=(self.dim1, self.dim2))
        if annotation == True:
            sn.heatmap(CMat, linewidths=.5, linecolor="black", xticklabels=labels,
                       yticklabels=labels, cmap=cMap, fmt=".4g", annot=annotation)
            ax.tick_params(axis="x", rotation=Rotations[0])
            ax.tick_params(axis="y", rotation=Rotations[1])
        else:
            sn.heatmap(CMat, linewidths=0., linecolor="black", xticklabels=labels,
                       yticklabels=labels, cmap=cMap, fmt=".4g", annot=annotation)
            ax.tick_params(bottom=False, top=False, left=False, right=False)
            ax.set_xticks([])
            ax.set_yticks([])
            
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
   
        accuracy, precision, recall, f1 = self.__metrics(yPred)
        Text = f"Accuracy :{accuracy}\nPrecision :{precision}\nRecall      :{recall}\nf1            :{f1}"
        fig.text(0.90,0.90,Text,fontsize=11)
        fig.suptitle(Title)
        
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig, savePlot[1], savePlot[2])
          
        return fig, ax