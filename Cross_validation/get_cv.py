#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Mar 1, 2020
@author: Chengning Zhang
"""

import warnings
warnings.filterwarnings("ignore")
def get_cv(cls,X,Y,M,n_splits=10,cv_type = "StratifiedKFold",verbose = True):  
  """ Cross validation to get CLL and accuracy and training time and precision and recall.
  """

  if cv_type == "StratifiedKFold":
    cv = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=42) # The folds are made by preserving the percentage of samples for each class.
  else: 
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
  
  model = cls()
  X,Y = check_X_y(X,Y)
  #binarizer = MultiLabelBinarizer() ## for using recall and precision score
  #binarizer.fit(Y)
  Accuracy = []
  Precision = []
  Recall = []
  CLL = []
  training_time = []
  F1 = []
  for folder, (train_index, val_index) in enumerate(cv.split(X, Y)): # X,Y are array, data is list
    X_train,X_val = X[train_index],X[val_index]
    y_train,y_val = Y[train_index],Y[val_index] 
    model.fit(X_train,y_train,M) # whether data is list or array does not matter, only thing matters is label has to be same.
    training_time.append(model.training_time_)
    Accuracy.append(accuracy_score(y_val, model.predict(X_val) ))
    CLL.append(model.Conditional_log_likelihood_general(y_val,model.predict_proba(X_val), model.classes_ ) )
    Precision.append(precision_score(y_val, model.predict(X_val), average='macro') )  
    Recall.append(recall_score(y_val, model.predict(X_val), average='macro') ) 
    F1.append(f1_score(y_val, model.predict(X_val), average='macro') ) 
    if verbose:
        print("accuracy in %s fold is %s" % (folder+1,Accuracy[-1] ) )
        print("CLL in %s fold is %s" % (folder+1,CLL[-1]))
        print("precision in %s fold is %s" % (folder+1,Precision[-1]))
        print("recall in %s fold is %s" % (folder+1,Recall[-1]))
        print("f1 in %s fold is %s" % (folder+1,F1[-1]))
        print("training time in %s fold is %s" % (folder+1,training_time[-1]))
        print(10*'__')
    
  return Accuracy, CLL, training_time,Precision,Recall,F1
