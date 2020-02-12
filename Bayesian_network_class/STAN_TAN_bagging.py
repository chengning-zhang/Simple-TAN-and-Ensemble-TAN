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

  
