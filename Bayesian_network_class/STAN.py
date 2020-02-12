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

