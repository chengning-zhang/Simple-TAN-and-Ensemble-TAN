
from Bayes_Network import Bayes_net
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
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted 


class NB(Bayes_net):
  name = "NB"
  def __init__(self, alpha = 1):
      self.alpha = alpha

  def fit(self,X, y):  
    """ Implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
      """

      #  countDict_, classes_, p_ , P_class_prior_, Dict_C_, K_ ,training_time_, is_fitted_  are fitted "coef_" 
       # coef_ has to been refreshed each fitting. 

    X, y = check_X_y(X, y)
    t = time.process_time()
    #start timing
    countDict = Counter(y) ## {c1:n1,c2:n2,c3:n3} sorted by counts
    C = list(countDict.keys()) ### [class1 , class2, class3] in appearing order
    n,p = X.shape   ## num of features 8                                                           ### values same order as .keys()
    P_class_prior = [(ele+self.alpha)/ ( n + self.alpha*len(C) )  for ele in countDict.values()]  ### prior for each class [p1,p2,p3]
    P_class_prior = dict(zip(C, P_class_prior))  ## {c1:p1,c2:p2,c3:p3} ## should in correct order, .keys .values.
    Dict_C = {} ###  {c1:[counter1, ....counter8], c2:[counter1, ....counter8],   c3: [counter1, ....counter8]}
    K = {} ## [x1 unique , x2 unique .... x8unique]

    for c in C:
      ListCounter_c = []

      for i in range(p):
        row_inx_c = [row for row in range(n) if y[row] == c]
        x_i_c = X[row_inx_c,i]
        ListCounter_c.append(Counter(x_i_c))
        if c == C[0]:
          x_i = X[:,i]
          K[i] = len(Counter(x_i))

      Dict_C[c] = ListCounter_c
    
    CP_time = np.array(time.process_time() - t) 
    self.is_fitted_ = True    
    self.Dict_C_,self.p_,self.P_class_prior_,self.K_,self.classes_,self.countDict_,self.training_time_ = Dict_C,p,P_class_prior,K,np.array(C),countDict,CP_time
    return self


  def predict_proba(self,X): 
    """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
    """
    check_is_fitted(self)
    X = check_array(X)
    Prob_C = []
    for ins in X:
      P_class = self.P_class_prior_.copy() ### {c1:p1, c2:p2} #### !!!! dict1 = dict2 , change both simultaneously!!!
      for c in self.classes_:
        ListCounter_c = self.Dict_C_[c]
        for i in range(self.p_):
          P_class[c] = P_class[c] * (ListCounter_c[i][ins[i]]+self.alpha) / (self.countDict_[c] + self.alpha*self.K_[i])
        
      ## normalize P_class
      P_class = {key: P_class[key]/sum(list(P_class.values())) for key in P_class.keys()}
      Prob_C.append(list(P_class.values())) ### check the class order is correct
  
    Prob_C = np.array(Prob_C) ### for shap !!!!
    return Prob_C

    
