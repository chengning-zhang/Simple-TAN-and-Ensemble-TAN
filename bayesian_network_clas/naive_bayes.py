class NB(Bayes_net):


  def get_name(self):
    return("NB")

  def fit(self,train):  
    Y = self.get_Y(train)
    t = time.process_time()
    """start timing"""
    countDict = Counter(Y) ## {c1:n1,c2:n2,c3:n3} sorted by counts
    C = list(countDict.keys()) ### [class1 , class2, class3] in appearing order
    p = len(train[0]) - 1    ## num of features 8                                  ### .values same order as .keys()
    P_class = [(ele+self.alpha)/(sum(list(countDict.values())) + self.alpha*len(C) )  for ele in countDict.values()]  ### prior for each class [p1,p2,p3]
    P_class = dict(zip(C, P_class))  ## {c1:p1,c2:p2,c3:p3} ## should in correct order, .keys .values.
    Dict_C = {} ###  {c1:[counter1, ....counter8], c2:[counter1, ....counter8],   c3: [counter1, ....counter8]}
    K = {} ## [x1 unique , x2 unique .... x8unique]

    for c in C:
      ListCounter_c = []

      for i in range(p):
        x_i_c = [ele[i] for ele in train if ele[-1] == c]
        ListCounter_c.append(Counter(x_i_c))
        if c == C[0]:
          x_i = [ele[i] for ele in train]
          K[i] = len(Counter(x_i))

      Dict_C[c] = ListCounter_c
    
    CP_time = time.process_time() - t; CP_time = np.array(CP_time)
    self._is_fitted = True
    self.Dict_C,self.p,self.P_class_prior,self.K,self.C,self.countDict,self.training_time = Dict_C,p,P_class,K,C,countDict,CP_time
    return self


  def predict(self,X_test): 
    """Predict prob values for test set for each class.
        :param test_set: Test set with dimension (p or p+1) x n,
                         where n is the number of examples,
                         and p is the number of features.
        :return: Predicted target values for test set with dimension n * |C|, 
                 where n is the number of examples. |C| is the # of classes. 
                 it is np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
        """
    if not self._is_fitted:
      raise NotFittedError(self.__class__.__name__)

    Prob_C = []
    for ins in X_test:
      P_class = self.P_class_prior.copy() ### {c1:p1, c2:p2} #### !!!! dict1 = dict2 , change both simultaneously!!!
      for c in self.C:
        ListCounter_c = self.Dict_C[c]
        for i in range(self.p):
          P_class[c] = P_class[c] * (ListCounter_c[i][ins[i]]+self.alpha) / (self.countDict[c] + self.alpha*self.K[i])
        
      ## normalize P_class
      P_class = {key: P_class[key]/sum(list(P_class.values())) for key in P_class.keys()}
      Prob_C.append(list(P_class.values())) ### check the class order is correct
  
    Prob_C = array(Prob_C) ### for shap !!!!
    return Prob_C

  def predict_binary(self,X_test):
    Prob_C = self.predict(X_test) ### Prob_C is n*|C| np.array
    return(Prob_C[:,0]) 
    
