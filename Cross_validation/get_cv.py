import warnings
warnings.filterwarnings("ignore")
def get_cv(cls,X,Y,M,n_splits=10,cv_type = "KFold",verbose = True):  
  """ Cross validation to get CLL and accuracy and training time and precision and recall.
  """

  if cv_type == "StratifiedKFold":
    cv = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=42)##The folds are made by preserving the percentage of samples for each class.
  else: 
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
  

  model = cls()
  X,Y = check_X_y(X,Y)
  binarizer = MultiLabelBinarizer() ## for using recall and precision score
  binarizer.fit(Y)
  Accuracy = []
  Precision = []
  Recall = []
  CLL = []
  training_time = []
  for folder, (train_index, val_index) in enumerate(cv.split(X, Y)):#### X,Y are array, data is list
    X_train,X_val = X[train_index],X[val_index]
    y_train,y_val = Y[train_index],Y[val_index] 
    model.fit(X_train,y_train,M) ### whether data is list or array does not matter, only thing matters is label has to be same.
    training_time.append(model.training_time_)

    y_pred_prob= model.predict_proba(X_val)  
    y_pred_class = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_class)
    precision = precision_score(binarizer.transform(y_val), 
         binarizer.transform(y_pred_class), 
         average='macro')    
    recall = recall_score(binarizer.transform(y_val), 
         binarizer.transform(y_pred_class), 
         average='macro')
    cll = model.Conditional_log_likelihood_general(y_val,y_pred_prob,model.classes_)
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
