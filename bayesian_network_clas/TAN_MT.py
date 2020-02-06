class TAN_MT(Bayes_net):
    def __init__(self, alpha = 1,starting_node = 0):
      self.starting_node = starting_node
      self.alpha = alpha
      self.name = "TAN_MT"

      self.Dict_C = []
      self.p = 0
      self.P_class_prior = []
      self.K = []
      self.C = 0
      self.countDict = []
      self.parent = [] ### one more attribute than NB

      self._is_fitted = False
      """add training time """
      self.training_time = 0
      self.mutual_inf_time = 0
      self.prim_time = 0
      self.CP_time = 0

    def To_CAT(self, X_i): 
      """For using CMI purpose, convert X_i e.g ['a','b','a']/['0','1','0']  to [0,1,0].
      :param X_i: one feature column. 
      :return: list(type int)
      """
      X_i_list = list(set(X_i));X_i_dict = dict(zip(X_i_list, arange(len(X_i_list)) ))
      return([X_i_dict[ele] for ele in X_i])

    def get_mutual_inf(self,train):
      """get conditional mutual inf of all pairs of features, part of training
      :return: np.array matrix.
      """
      t = time.process_time()
      p = len(train[0]) - 1
      M = np.zeros((p,p))
      Y = self.get_Y(train); Y = self.To_CAT(Y)
      X = self.get_X(train)
      for i in range(p):
        X_i = [ele[i] for ele in X]
        X_i = self.To_CAT(X_i)
        for j in range(p):
          X_j = [ele[j] for ele in X]; 
          X_j = self.To_CAT(X_j)
          M[i,j] = drv.information_mutual_conditional(X_i,X_j,Y)
      
      self.mutual_inf_time = time.process_time() - t
      return M

    def Findparent(self,train):
      M = self.get_mutual_inf(train)
      t = time.process_time()
      fill_diagonal(M,0)  
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
      
      self.prim_time = time.process_time() - t
      return parent

    def fit(self,train):  ### this is based on trainning data !!!
      parent = self.Findparent(train)
      y = self.get_Y(train)
      t = time.process_time()
      """ start timing"""
      countDict = Counter(y)
      C = list(countDict.keys()) ### [class1 , class2, class3] in appearing order
      p = len(train[0]) - 1
      P_class = [(ele+self.alpha)/(sum(list(countDict.values())) + self.alpha*len(C) )  for ele in list(countDict.values())]  ### prior for each class [p1,p2,p3], ### .values same order as .keys()
      P_class = dict(zip(C, P_class))  ## {c1:p1,c2:p2,c3:p3} ## should in correct order, .keys .values.
      Dict_C = {} ###  {c1:[counter1, ....counter8], c2:[counter1, ....counter8],   c3: [counter1, ....counter8]}
      K = {}

      root_i = self.starting_node ## 0 ,1 ,2 shows the position, thus int
      x_i = [ele[root_i] for ele in train]
      K[root_i] = len(Counter(x_i))
      for c in C: ### c origianl class label '1'   not 1
        ListCounter_c = {}
        x_i_c = [ele[root_i] for ele in train if ele[-1] == c]
        ListCounter_c[root_i] = Counter(x_i_c) ### list_counter_c keys are 0,1,2,3... showing position hence int. Counter(x_i_c) keys are original values of x, not position. hence not necesarily int
        for i in [e for e in range(0,p) if e != root_i]:
          if c == C[0]:
            x_i = [ele[i] for ele in train]
            K[i] =len(Counter(x_i))
          x_parent = [ele[parent[i]] for ele in train] ## will duplicate C times. 
          x_parent_counter = Counter(x_parent)
          x_parent_counter_length = len(x_parent_counter)
          x_parent_value = list(x_parent_counter.keys())
          dict_i_c = {}
          for j in range(x_parent_counter_length):
            x_i_c_p_j = [ele[i] for ele in train if ele[-1] == c and ele[parent[i]] == x_parent_value[j] ]
            dict_i_c[x_parent_value[j]] = Counter(x_i_c_p_j) ### x_parent_value[j] can make sure it is right key.
          ListCounter_c[i] = dict_i_c
        Dict_C[c] = ListCounter_c 

      CP_time = time.process_time() - t
      self._is_fitted = True
      self.Dict_C,self.p,self.P_class_prior,self.K,self.C,self.countDict, self.parent,self.CP_time = Dict_C,p,P_class,K,C,countDict,parent,CP_time
      self.training_time = np.array([self.mutual_inf_time,self.prim_time,self.CP_time])
      return self

    def predict(self,test):	
      """Predict prob values for test set for each class.
        :param test_set: Test set with dimension (p or p+1) x n,
                         where n is the number of examples,
                         and p is the number of features.
        :return: Predicted target values for test set with dimension n * |C|, 
                 where n is the number of examples. |C| is the # of classes. 
                 it is np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
      """
      if not self._is_fitted:
        raise NotFittedError(self.__class__.__name__) ### after fitting, self.Dict_C,self.p,self.P_class_prior,self.K,self.C,self.countDict, self.parent

      Prob_C = []
      root_i = self.starting_node

      for ins in test:
        P_class = self.P_class_prior.copy()
        for c in self.C:
          ListCounter_c = self.Dict_C[c]
          P_class[c] = P_class[c] * (ListCounter_c[root_i][ins[root_i]]+self.alpha) / (self.countDict[c]+self.alpha*self.K[root_i])
        
          for i in [e for e in range(0,self.p) if e != root_i]:
            pValue = ins[self.parent[i]] ### replicate C times
            try:###  ListCounter_c[i][pValue],pavlue does show in training
              Deno = sum(list(ListCounter_c[i][pValue].values() )) ## number of y =1, xparent = pvalue ,   ListCounter_c[i][pValue], pavlue does not show in training , keyerror
              P_class[c] = P_class[c] * (ListCounter_c[i][pValue][ins[i]] + self.alpha) / (Deno + self.alpha*self.K[i]) ## ListCounter1[i][pValue][ins[i]] = number of y =1 xparent = pvalue, xi = xi
            except: ##ListCounter_c[i][pValue],pavlue does not show in training
              Deno = 0 ## ListCounter_c[i] this is when class == c, ith feature,  >> {parent(i) == value1: Counter,  parent(i) == value2: Counter  },  counter shows the distribution of x_i when class ==c and parent == pvalue
              P_class[c] = P_class[c] * (0 + self.alpha) / (Deno + self.alpha*self.K[i])
        
        P_class = {key: P_class[key]/sum(list(P_class.values())) for key in P_class.keys()} ### normalize p_class
        Prob_C.append(list(P_class.values())) ### check the class order is correct

      Prob_C = array(Prob_C) ### for shap !!!!
      return Prob_C

    def predict_binary(self,test):
      Prob_C = self.predict(test)
      return(Prob_C[:,0]) 

