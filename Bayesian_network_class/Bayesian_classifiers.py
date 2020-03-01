#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Mar 1, 2020
@author: Chengning Zhang
"""

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
import networkx as nx
import matplotlib.pyplot as plt 

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted ### Checks if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing underscore)
#from sklearn.utils.multiclass import unique_labels, not necessary, can be replaced by array(list(set()))





class Bayes_net(BaseEstimator, ClassifierMixin): 
    """
    Bayesian network implementation
    API inspired by SciKit-learn.
    """
    def fit(self,X,y,M = None):
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
      :param C: Class labels  e.x array(['1','0']), C has to use same labels as y_true.
      :return: CLL. A scalar.
      """
      C = list(C) ## only list can use .index
      cll = []
      for i in range(len(y_true)):
        cll.append( y_pred_prob[i,C.index(y_true[i])] ) ## \hat p(c_true|c_true)
      
      cll = [np.log2(ele) for ele in cll]
      cll = np.array(cll)
      return(sum(cll))
 
    def plot_tree_structure(self,mapping = None,figsize = (5,5)):
      check_is_fitted(self)
      parent = self.parent_
      egdes = [(k,v) for v,k in parent.items() if k is not None]
      G = nx.MultiDiGraph()
      G.add_edges_from(egdes)
      #mapping=dict(zip(range(8),['b0','b1','b2','b3','b4','b5','b6','b7']))
      plt.figure(figsize=figsize)
      nx.draw_networkx(G,nx.shell_layout(G))


class NB(Bayes_net):
  name = "NB"
  def __init__(self, alpha = 1):
      self.alpha = alpha

  def fit(self,X, y, M = None):  
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


class TAN(Bayes_net):
    name = "TAN"
    def __init__(self, alpha = 1,starting_node = 0):
      self.starting_node = starting_node
      self.alpha = alpha

    def To_CAT(self, X_i): 
      """For using CMI purpose, convert X_i e.g ['a','b','a']/['0','1','0']  to [0,1,0].
      :param X_i: one feature column. 
      :return: list(type int)
      """
      X_i_list = list(set(X_i));X_i_dict = dict(zip(X_i_list, range(len(X_i_list)) ))
      return([X_i_dict[ele] for ele in X_i])

    def get_mutual_inf(self,X,Y):
      """get conditional mutual inf of all pairs of features, part of training
      :return: np.array matrix.
      """
      t = time.process_time()
      n,p = X.shape 
      M = np.zeros((p,p))
      Y = self.To_CAT(Y)
      for i in range(p):
        X_i = X[:,i]
        X_i = self.To_CAT(X_i)
        for j in range(p):
          X_j = X[:,j] 
          X_j = self.To_CAT(X_j)
          M[i,j] = drv.information_mutual_conditional(X_i,X_j,Y)
      
      mutual_inf_time = time.process_time() - t
      return M, mutual_inf_time

    def Findparent(self,X,Y):
      M,mutual_inf_time = self.get_mutual_inf(X,Y)
      t = time.process_time()
      np.fill_diagonal(M,0)  
      p = int(M.shape[0])  
      V = range(p)  #### . set of all nodes
      st = self.starting_node
      Vnew = [st] #### vertex that already found their parent. intitiate it with starting node. TAN randomly choose one
      parent = {st:None} ## use a dict to show nodes' interdepedency
      while set(Vnew) != set(V):   ### when their are still nodes whose parents are unknown.
        index_i = [] ### after for loop, has same length as Vnew, shows the closest node that not in Vnew with Vnew.  
        max_inf = [] ### corresponding distance
        for i in range(len(Vnew)):  ## can be paralelled 
          vnew = Vnew[i]
          ListToSorted = [int(e) for e in M[:,vnew]]###
          index = sorted(range(len(ListToSorted)),key = lambda k: ListToSorted[k],reverse = True)
          index_i.append([ele for ele in index if ele not in Vnew][0]) 
          max_inf.append(M[index_i[-1],vnew])
      
        index1 = sorted(range(len(max_inf)),key = lambda k: max_inf[k],reverse = True)[0] ## relative position, Vnew[v1,v2] index_i[v4,v5] max_inf[s1,s2] index1 is the position in those 3 list
        Vnew.append(index_i[index1]) ### add in that node
        parent[index_i[index1]] = Vnew[index1] ## add direction, it has to be that the new added node is child, otherwise some nodes has 2 parents which is wrong.
      
      prim_time = time.process_time() - t
      return parent,mutual_inf_time,prim_time

    def fit(self,X,y,M = None):  ### this is based on trainning data !!!
      X, y = check_X_y(X, y)

      parent,mutual_inf_time,prim_time = self.Findparent(X,y)
      t = time.process_time()
      countDict = Counter(y)
      C = list(countDict.keys()) ### [class1 , class2, class3] in appearing order
      n,p = X.shape
      P_class = [(ele+self.alpha)/( n + self.alpha*len(C) )  for ele in list(countDict.values())]  ### prior for each class [p1,p2,p3], ### .values same order as .keys()
      P_class = dict(zip(C, P_class))  ## {c1:p1,c2:p2,c3:p3} ## should in correct order, .keys .values.
      Dict_C = {} ###  {c1:[counter1, ....counter8], c2:[counter1, ....counter8],   c3: [counter1, ....counter8]}
      K = {}

      root_i = self.starting_node ## 0 ,1 ,2 shows the position, thus int
      x_i = X[:,root_i]
      K[root_i] = len(Counter(x_i))
      for c in C: ### c origianl class label '1'   not 1
        ListCounter_c = {}
        row_inx_c = [row for row in range(n) if y[row] == c]
        x_i_c = X[row_inx_c,root_i]
        ListCounter_c[root_i] = Counter(x_i_c) ### list_counter_c keys are 0,1,2,3... showing position hence int. Counter(x_i_c) keys are original values of x, not position. hence not necesarily int
        for i in [e for e in range(0,p) if e != root_i]:
          if c == C[0]:
            x_i = X[:,i]
            K[i] =len(Counter(x_i))
          x_parent = X[:,parent[i]]
          x_parent_counter = Counter(x_parent)
          x_parent_counter_length = len(x_parent_counter)
          x_parent_value = list(x_parent_counter.keys())
          dict_i_c = {}
          for j in range(x_parent_counter_length):
            row_inx_c_parent_j = [row for row in range(n) if y[row] == c and x_parent[row] == x_parent_value[j]]
            x_i_c_p_j = X[row_inx_c_parent_j, i]
            dict_i_c[x_parent_value[j]] = Counter(x_i_c_p_j) ### x_parent_value[j] can make sure it is right key.
          ListCounter_c[i] = dict_i_c
        Dict_C[c] = ListCounter_c 

      CP_time = time.process_time() - t
      self.is_fitted_ = True 
      self.Dict_C_,self.p_,self.P_class_prior_,self.K_,self.classes_,self.countDict_, self.parent_ = Dict_C,p,P_class,K,np.array(C),countDict,parent
      self.training_time_ = np.array([mutual_inf_time,prim_time,CP_time])
      return self

    def predict_proba(self,X):	
      check_is_fitted(self)
      X = check_array(X)

      Prob_C = []
      root_i = self.starting_node

      for ins in X:
        P_class = self.P_class_prior_.copy()
        for c in self.classes_:
          ListCounter_c = self.Dict_C_[c]
          P_class[c] = P_class[c] * (ListCounter_c[root_i][ins[root_i]]+self.alpha) / (self.countDict_[c]+self.alpha*self.K_[root_i])
        
          for i in [e for e in range(0,self.p_) if e != root_i]:
            pValue = ins[self.parent_[i]] ### replicate C times
            try:###  ListCounter_c[i][pValue],pavlue does show in training
              Deno = sum(list(ListCounter_c[i][pValue].values() )) ## number of y =1, xparent = pvalue ,   ListCounter_c[i][pValue], pavlue does not show in training , keyerror
              P_class[c] = P_class[c] * (ListCounter_c[i][pValue][ins[i]] + self.alpha) / (Deno + self.alpha*self.K_[i]) ## ListCounter1[i][pValue][ins[i]] = number of y =1 xparent = pvalue, xi = xi
            except: ##ListCounter_c[i][pValue],pavlue does not show in training
              Deno = 0 ## ListCounter_c[i] this is when class == c, ith feature,  >> {parent(i) == value1: Counter,  parent(i) == value2: Counter  },  counter shows the distribution of x_i when class ==c and parent == pvalue
              P_class[c] = P_class[c] * (0 + self.alpha) / (Deno + self.alpha*self.K_[i])
        
        P_class = {key: P_class[key]/sum(list(P_class.values())) for key in P_class.keys()} ### normalize p_class
        Prob_C.append(list(P_class.values())) ### check the class order is correct

      Prob_C = np.array(Prob_C) ### for shap !!!!
      return Prob_C
      
class STAN(Bayes_net):
    name = "STAN"
    def __init__(self,alpha = 1,starting_node = 0):
      self.starting_node = starting_node
      self.alpha = alpha

    def Findparent(self,M):
      M = M.copy()
      np.fill_diagonal(M,0)  
      p = int(M.shape[0])  
      V = range(p)  #### . set of all nodes
      st = self.starting_node
      Vnew = [st] #### vertex that already found their parent. intitiate it with starting node. TAN randomly choose one
      parent = {st:None} ## use a dict to show nodes' interdepedency
      while set(Vnew) != set(V):   ### when their are still nodes whose parents are unknown.
        index_i = [] ### after for loop, has same length as Vnew, shows the closest node that not in Vnew with Vnew.  
        max_inf = [] ### corresponding distance
        for i in range(len(Vnew)):  ## can be paralelled 
          vnew = Vnew[i]
          ListToSorted = [int(e) for e in M[:,vnew]]###
          index = sorted(range(len(ListToSorted)),key = lambda k: ListToSorted[k],reverse = True)
          index_i.append([ele for ele in index if ele not in Vnew][0]) 
          max_inf.append(M[index_i[-1],vnew])
      
        index1 = sorted(range(len(max_inf)),key = lambda k: max_inf[k],reverse = True)[0] ## relative position, Vnew[v1,v2] index_i[v4,v5] max_inf[s1,s2] index1 is the position in those 3 list
        Vnew.append(index_i[index1]) ### add in that node
        parent[index_i[index1]] = Vnew[index1] ## add direction, it has to be that the new added node is child, otherwise some nodes has 2 parents which is wrong.
      
      return parent

    def fit(self,X,y,M):  ### this is based on trainning data !!!
      X, y = check_X_y(X, y)
      parent = self.Findparent(M)
      t = time.process_time()
      countDict = Counter(y)
      C = list(countDict.keys()) ### [class1 , class2, class3] in appearing order
      n,p = X.shape
      P_class = [(ele+self.alpha)/( n + self.alpha*len(C) )  for ele in list(countDict.values())]  ### prior for each class [p1,p2,p3], ### .values same order as .keys()
      P_class = dict(zip(C, P_class))
      Dict_C = {} ###  {c1:[counter1, ....counter8], c2:[counter1, ....counter8],   c3: [counter1, ....counter8]}
      K = {}

      root_i = self.starting_node ## 0 ,1 ,2 shows the position, thus int
      x_i = X[:,root_i]
      K[root_i] = len(Counter(x_i))
      for c in C: ### c origianl class label '1'   not 1
        ListCounter_c = {}
        row_inx_c = [row for row in range(n) if y[row] == c]
        x_i_c = X[row_inx_c,root_i]
        ListCounter_c[root_i] = Counter(x_i_c) ### list_counter_c keys are 0,1,2,3... showing position hence int. Counter(x_i_c) keys are original values of x, not position. hence not necesarily int
        for i in [e for e in range(0,p) if e != root_i]:
          if c == C[0]:
            x_i = X[:,i]
            K[i] =len(Counter(x_i))
          x_parent = X[:,parent[i]] ## will duplicate C times. 
          x_parent_counter = Counter(x_parent)
          x_parent_counter_length = len(x_parent_counter)
          x_parent_value = list(x_parent_counter.keys())
          dict_i_c = {}
          for j in range(x_parent_counter_length):
            row_inx_c_parent_j = [row for row in range(n) if y[row] == c and x_parent[row] == x_parent_value[j]]
            x_i_c_p_j = X[row_inx_c_parent_j, i]
            dict_i_c[x_parent_value[j]] = Counter(x_i_c_p_j) ### x_parent_value[j] can make sure it is right key.
          ListCounter_c[i] = dict_i_c
        Dict_C[c] = ListCounter_c 

      CP_time = np.array(time.process_time() - t)
      self.is_fitted_ = True
      self.Dict_C_,self.p_,self.P_class_prior_,self.K_,self.classes_,self.countDict_,self.parent_ = Dict_C,p,P_class,K,np.array(C),countDict,parent
      self.training_time_ = CP_time
      return self


    def predict_proba(self,X):	
      check_is_fitted(self)
      X = check_array(X)

      Prob_C = []
      root_i = self.starting_node

      for ins in X:
        P_class = self.P_class_prior_.copy()
        for c in self.classes_:
          ListCounter_c = self.Dict_C_[c]
          P_class[c] = P_class[c] * (ListCounter_c[root_i][ins[root_i]]+self.alpha) / (self.countDict_[c]+self.alpha*self.K_[root_i])
        
          for i in [e for e in range(0,self.p_) if e != root_i]:
            pValue = ins[self.parent_[i]] ### replicate C times
            try:###  ListCounter_c[i][pValue],pavlue does show in training
              Deno = sum(list(ListCounter_c[i][pValue].values() )) ## number of y =1, xparent = pvalue ,   ListCounter_c[i][pValue], pavlue does not show in training , keyerror
              P_class[c] = P_class[c] * (ListCounter_c[i][pValue][ins[i]] + self.alpha) / (Deno + self.alpha*self.K_[i]) ## ListCounter1[i][pValue][ins[i]] = number of y =1 xparent = pvalue, xi = xi
            except: ##ListCounter_c[i][pValue],pavlue does not show in training
              Deno = 0 ## ListCounter_c[i] this is when class == c, ith feature,  >> {parent(i) == value1: Counter,  parent(i) == value2: Counter  },  counter shows the distribution of x_i when class ==c and parent == pvalue
              P_class[c] = P_class[c] * (0 + self.alpha) / (Deno + self.alpha*self.K_[i])
        
        P_class = {key: P_class[key]/sum(list(P_class.values())) for key in P_class.keys()} ### normalize p_class
        Prob_C.append(list(P_class.values())) ### check the class order is correct

      Prob_C = np.array(Prob_C) ### for shap !!!!
      return Prob_C



class TAN_bagging(Bayes_net):
  name = "TAN_bagging"
  def __init__(self, alpha = 1):
    self.alpha = alpha

  def fit(self,X,y,M = None): 
    """initialize model = [] . and training time."""
    X,y = check_X_y(X,y)
    n,p = X.shape ### number of features
    """fit base models"""
    training_time = 0
    models = []
    for i in range(p):
      model = TAN(self.alpha, starting_node= i)
      model.fit(X,y)
      models.append(model)
      training_time += model.training_time_

    self.models_ , self.p_= models,p
    self.training_time_ = training_time/p ### the fitting can be paralelled, hence define averge training time for this bagging
    self.is_fitted_ = True
    self.classes_ = model.classes_
    return self

  def predict_proba(self,X):	   
    check_is_fitted(self)
    X = check_array(X)

    Prob_C = 0
    for model in self.models_:
      Prob_C += model.predict_proba(X) ### get np array here 

    Prob_C = Prob_C/self.p_
    return(Prob_C)



class STAN_bagging(Bayes_net):
  name = "STAN_bagging"
  def __init__(self,alpha = 1):
    self.alpha = alpha

  def fit(self,X,y,M): 
    X,y = check_X_y(X,y)
    n,p = X.shape
    training_time = 0
    models = []
    for i in range(p):
      model = STAN(self.alpha, starting_node= i)
      model.fit(X,y,M)
      models.append(model)
      training_time += model.training_time_

    self.models_, self.p_ = models,p
    self.training_time_ = training_time/p ### the fitting can be paralelled, hence define averge training time for this bagging
    self.is_fitted_ = True
    self.classes_ = model.classes_
    return self

  def predict_proba(self,X):	   
    check_is_fitted(self)
    X = check_array(X)

    Prob_C = 0
    for model in self.models_:
      Prob_C += model.predict_proba(X) ### get np array here 

    Prob_C = Prob_C/self.p_
    return(Prob_C)


class STAN_TAN_bagging(Bayes_net):
  name = "STAN_TAN_bagging"
  def __init__(self,alpha = 1):
    self.alpha = alpha

  def fit(self,X,y,M): 
    X,y = check_X_y(X,y)
    n,p = X.shape  
    training_time = 0
    models = []
    ## train p TAN base models
    for i in range(p):
      model = TAN(self.alpha, starting_node= i)
      model.fit(X,y)
      models.append(model)
      training_time += model.training_time_

    #append STAN
    model = STAN(self.alpha, starting_node = 0) ### starting node not importance for TAN, very robust
    model.fit(X,y,M)
    models.append(model)    
    self.models_, self.p_ = models, p
    self.training_time_ = training_time/p ### after paralell, only consider average of p TAN_MT, ignore TAN since it takes less time than TAN_MT
    self.is_fitted_ = True
    self.classes_ = model.classes_
    return self

  def predict_proba(self,X):	   
    check_is_fitted(self)
    X = check_array(X)

    Prob_C = 0
    for model in self.models_:
      Prob_C += model.predict_proba(X) ### get np array here 

    Prob_C = Prob_C/(self.p_+ 1)
    return(Prob_C)

  






