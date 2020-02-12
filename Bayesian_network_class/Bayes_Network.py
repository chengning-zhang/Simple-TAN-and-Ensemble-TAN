"""
Bayesian network implementation
API inspired by SciKit-learn.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted ### Checks if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing underscore)
#from sklearn.utils.multiclass import unique_labels, not necessary, can be replaced by array(list(set()))
from __future__ import division ###for float operation
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score ##tp / (tp + fn)
from sklearn.metrics import precision_score #tp / (tp + fp)
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import KFold, StratifiedKFold
from pyitlib import discrete_random_variable as drv
import time 
import timeit 

class Bayes_net(BaseEstimator, ClassifierMixin): 
    
    def fit(self,X,y):
      raise NotImplementedError

    def predict_proba(self, X): ### key prediction methods, all other prediction methods will use it first.
      raise NotImplementedError

    def predict_binary(self,X):
      """
        Perform classification on an array of test vectors X, predict P(C1|X), works only for binary classifcation
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted P(C1|X)
      """
      Prob_C = self.predict_proba(X) ### Prob_C is n*|C| np.array
      return(Prob_C[:,0]) 

    def predict(self, X):
      """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X
      """
      Prob_C = self.predict_proba(X) ## Prob_C is |C|*n np.array ,C is self.C 
      return( np.array([self.classes_[ele] for ele in np.argmax(Prob_C, axis=1)] ) )

    def Conditional_log_likelihood_general(self,y_true,y_pred_prob,C): 
      """Calculate the conditional log likelihood.
      :param y_true: The true class labels. e.g ['1','1',.....'0','0']
      :param y_pred_prob: np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
      :param C: Class labels  e.x ['1','0'], C has to use same labels as y_true.
      :return: CLL. A scalar.
      """
      cll = []
      for i in range(len(y_true)):
        cll.append( y_pred_prob[i,C.index(y_true[i])] ) ## \hat p(c_true|c_true)
      
      cll = [np.log2(ele) for ele in cll]
      cll = np.array(cll)
      return(sum(cll))
 
