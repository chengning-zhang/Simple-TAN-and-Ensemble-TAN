def get_cv(model,data,n_splits=10,cv_type = "KFold",verbose = True):  
  """ Cross validation to get CLL and accuracy and training time.
    :param data: data with dimension p+1 x n, to be cross validated.
                         where n is the number of examples,
                         and p is the number of features. Last column is target.   data has to be a list [[ ], [ ], [ ].... ]
    :return CLL, accuracy, training time for each folds.
  """

  if cv_type == "StratifiedKFold":
    cv = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=42)##The folds are made by preserving the percentage of samples for each class.
  else: 
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
  
  X = np.array(model.get_X(data))## if data is array, then get_X return array. if data is list, get_X return list
  Y = np.array(model.get_Y(data)) ### Y array['1','0','']
  binarizer = MultiLabelBinarizer() ## for using recall and precision score
  binarizer.fit(Y)
  Accuracy = []
  Precision = []
  Recall = []
  CLL = []
  training_time = []
  for folder, (train_index, val_index) in enumerate(cv.split(X, Y)):#### X,Y are array, data is list
    X_val = X[val_index]
    y_val = Y[val_index] 
    model.fit(data[train_index]) ### whether data is list or array does not matter, only thing matters is label has to be same.
    y_pred_prob= model.predict(X_val)
    training_time.append(model.training_time)  
    y_pred_class = model.prob_to_class_general(y_pred_prob,model.C)
    accuracy = accuracy_score(y_val, y_pred_class)
    precision = precision_score(binarizer.transform(y_val), 
         binarizer.transform(y_pred_class), 
         average='macro')    
    recall = recall_score(binarizer.transform(y_val), 
         binarizer.transform(y_pred_class), 
         average='macro')
    cll = model.Conditional_log_likelihood_general(y_val,y_pred_prob,model.C)
    if verbose:
        print("accuracy in %s fold is %s" % (folder+1,accuracy))
        print("CLL in %s fold is %s" % (folder+1,cll))
        print("precision in %s fold is %s" % (folder+1,precision))
        print("recall in %s fold is %s" % (folder+1,recall))
        print("training time in %s fold is %s" % (folder+1,training_time[-1]))
        print(10*'__')
    CLL.append(cll)
    Accuracy.append(accuracy)
    Recall.append(recall)
    Precision.append(precision)
  return Accuracy, CLL, training_time,Precision,Recall
