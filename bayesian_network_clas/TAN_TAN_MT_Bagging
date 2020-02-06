class TAN_TAN_MT_bagging(Bayes_net):
  def __init__(self,Matrix,alpha = 1):
    self.alpha = alpha
    self.name = "TAN_TAN_MT_bagging"
    self.M = Matrix

    """base models, each having different starting node"""
    self.models = []
    self.p = 0
    self.C = []
    self._is_fitted = False
    """add training time """
    self.training_time = 0
    

  def fit(self,train): 
    """initialize models = [] . and training time."""
    p = len(train[0]) -1 ### number of features
    self.p = p
    """fit base models"""
    training_time = 0
    models = []
    for i in range(p):
      model = TAN_MT(self.alpha, starting_node= i)
      model.fit(train)
      models.append(model)
      training_time += model.training_time

    """append TAN"""
    model = TAN(self.M,self.alpha, starting_node= 0) ### starting node not importance for TAN, very robust
    model.fit(train)
    models.append(model)    
    self.models = models
    self.training_time = training_time/p ### only consider average of p TAN_MT, ignore TAN since it takes less time than TAN_MT
    self._is_fitted = True
    self.C = model.C
    return self

  def predict(self,test):	   
    if not self._is_fitted:
        raise NotFittedError(self.__class__.__name__) ### after fitting, self.Dict_C,self.p,self.P_class_prior,self.K,self.C,self.countDict, self.parent

    Prob_C = 0
    for model in self.models:
      Prob_C += model.predict(test) ### get np array here 

    Prob_C = Prob_C/(self.p+1)
    return(Prob_C)

  def predict_binary(self,test):
      Prob_C = self.predict(test)
      return(Prob_C[:,0]) 
  
