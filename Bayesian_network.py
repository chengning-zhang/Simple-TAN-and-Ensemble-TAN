"""
Bayesian network implementation
API inspired by SciKit-learn.
"""

class Bayes_net(object): ### DO I need object???
    def __init__(self, alpha = 1):
      """Create a Bayesian classifier
      alpha is the smoothing parameter
      name is the class name of that classifier: Naive Bayes, Tree augmented naive Nayes
      Dict_C, p , P_class_prior,C K,countDict are learned from fitting method. Initialized as empty list.
      """
      self.alpha = alpha
      self.name = self.get_name()

      self.Dict_C = []
      self.p = 0
      self.P_class_prior = []
      self.K = []
      self.C = []
      self.countDict = []
      self._is_fitted = False

      """add training time """
      self.training_time = 0


    def get_name(self):
      raise NotImplementedError

    def get_Y(self,train):  
      """Get target values from train data.
      :param train: Training examples with dimension (p+1) x n,
                    where n is the number of examples,
                    and p is the number of features. 
                    Last column is target.(dtype: list)
      :return: target values(dtype: list/array)
      """
      if isinstance(train,list):
        return([ele[-1] for ele in train])
      else: ### suppose it is array
        return(train[:,-1])

    def get_X(self,train):
      """Get feature values from train data.
      :param train: Training examples with dimension (p+1) x n,
                    where n is the number of examples,
                    and p is the number of features. 
                    Last column is target.(dtype: list)
      :return: feature matrix(dtype: list[[ ],[ ],[ ],....[ ]] or array) 
      """
      if isinstance(train,list):
        p = len(train[0]) - 1
        return([ele[0:p] for ele in train])
      else:
        p = len(train[0]) - 1
        return(train[:,0:p])

    def prob_to_class_general(self,y_pred_prob,C): 
      """convert predicted probabilities to class labels.
      :param y_pred_prob: np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
      :param C: Class labels 
      :return: predicted class labels.(dtype:array).    if C = ['1','0'], then returns list like array(['1','1','0'])
      """

      return( np.array([C[ele] for ele in np.argmax(y_pred_prob, axis=1)] ) ) 

    def Conditional_log_likelihood_general(self,y_true,y_pred_prob,C): 
      """Calculate the conditional log likelihood.
      :param y_true: The true class labels. e.g ['1','1',.....'0','0']
      :param y_pred_prob: np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
      :param C: Class labels  e.x ['1','0'], C has to use same labels as y_true.
      :return: CLL. A scalar.
      """
      cll = []
      for i in range(len(y_true)):
        cll.append( y_pred_prob[i,C.index(y_true[i])] ) ## \hat p(c_true|c_true)
      
      cll = [np.log2(ele) for ele in cll]
      cll = np.array(cll)
      return(sum(cll))

    def fit(self,train):
      """Reset the parameters to none, and Fit model according to train data.
        :param train: Training examples with dimension (p+1) x n,
                      where n is the number of examples,
                      and p is the number of features. 
                      Last column is target.(dtype: list)
        :return: self
      """
      raise NotImplementedError
    
    def predict(self, test):
      """Predict prob values for test set for each class.
        :param test_set: Test set with dimension (p or p+1) x n,
                         where n is the number of examples,
                         and p is the number of features.
        :return: Predicted target values for test set with dimension n * |C|, 
                 where n is the number of examples. |C| is the # of classes. 
                 it is np.array shows prob of each class for each instance. ith column is the predicted prob for class C[i]
        """
      raise NotImplementedError
    
    def predict_binary(self,test):
      raise NotImplementedError

    def predict_class(self, test):
      """Predict class labels for test set .
        :param test_set: Test set with dimension (p or p+1) x n,
                         where n is the number of examples,
                         and p is the number of features.
        :return: Predicted class labels for test set with dimension n ,
                 where n is the number of examples.  
        """
      Prob_C = self.predict(test) ## Prob_C is |C|*n np.array ,C is self.C 
      return(self.prob_to_class_general(Prob_C,self.C))


 
