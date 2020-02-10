# Simpler TAN + Ensemble TAN
A custom implementation of Bayesian network written from scratch in Python 3, API inspired by SciKit-learn.


## Naive Bayes
From [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier):

> In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.


## TAN
```
_____________________________________________
Algorithm TAN(D)
---------------------------------------------
Input: a training instance set D
Output: the built TAN
1. compute the conditional mutual information I(A_i;A_j|C) between each pair of attributes, i \neq j
2. Build a complete undirected graph in which nodes are attributes A_1,...A_m. Annotate the weight of an edge connecting A_i to A_j by I(A_i;A_j|C).
3. Build a complete undirected maximum weighted spanning tree.
4. Transform the built undirected tree to a directed one by randomly choosing a root attribute and setting the direction of all edges to be outward from it. 
5. Build a TAN model by adding a node labeled by C and adding an arc from C to each A_i
6. Return the built TAN
```

TAN estimate class probability using:
P(C|a_1,...a_m) = \frac{ P(C). P(root|C).\prod_{j \neq root} P(a_j|C,parent(a_j) ) } { P(a_1,...a_m)}


## Simpler TAN
```
_____________________________________________
Algorithm Simpler TAN(D,M)
---------------------------------------------
Input: a training instance set D, contact matrix M
Output: the built Simpler TAN
1. Obtain the contact matrix M from domain expert.
2. Build a complete undirected graph in which nodes are attributes A_1,...A_m. Annotate the weight of an edge connecting A_i to A_j by M_{i,j}.
3. Build a complete undirected maximum weighted spanning tree.
4. Transform the built undirected tree to a directed one by randomly choosing a root attribute and setting the direction of all edges to be outward from it. 
5. Build a Simpler TAN model by adding a node labeled by C and adding an arc from C to each A_i
6. Return the built model
```

Simpler TAN estimate class probability using:

P(C|a_1,...a_m) = \frac{ P(C). P(root|C).\prod_{j \neq root} P(a_j|C,parent(a_j) ) } { P(a_1,...a_m)}



## Ensemble TAN
```
_____________________________________________
Algorithm Ensemble TAN training (D,M)
---------------------------------------------
Input: a training instance set D, contact matrix M.
Output: Simpler TAN, TAN1,TAN2,...TANm
For Simpler TAN:
   1. Obtain the contact matrix M from domain expert.
   2. Run Simpler TAN(D,M)
   3. Return Simpler TAN
For TAN1,....TANm:
  1. Compute the conditional mutual information I(A_i;A_j|C) between each pair of attributes, i \neq j.
  2. For each attribute A_i(i = 1,2,...m):
     (a) Choose A_i as the root node to build a complete directed maximum weighted spanning tree. Annotate the weight of an            edge connecting A_i to A_j by I(A_i;A_j|C).
     (b) Build TAN_i by adding a node labeled by C and adding an arc from C to each attribute A_i.

  3. Return the built TAN_1,...TAN_m
  
Return Simpler TAN, TAN_1,....TAN_m
``` 


```
_____________________________________________
Algorithm Ensemble TAN test(Simpler TAN, TAN_1,...TAN_m,e)
---------------------------------------------
Input: The built Simpler TAN, TAN_1,...TAN_m and a test instance e
Output: The probabilities \hat{P(c1|e)},.... \hat{P(c_C|e)}
1. For each class label c_p (p = 1,2,....C)
    (a) For each base classifier, use each one to estimate the probability \hat{P(c_p|e)} that e belongs to class c_p
    (b) Average all of the probabilities \hat{P(c_p|e)} = \frac{1}{m+1} \sum_0^m \hat{P((c_p|e))}
2. return the estimated \hat{P(c1|e)},.... \hat{P(c_C|e)}
```



## SHAP

![P450_shap_NB](/P450_nb_shap.png)

Format: ![Alt Text](url)



