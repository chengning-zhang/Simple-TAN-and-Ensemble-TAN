# Simpler-Tree-Augmented-Naive-Bayes
Bayesian network implementation API inspired by SciKit-learn.

# Incorporating Contact Matrix Information Using Bayesian Network Into Protein Sequence Analysis 

## Abstract:
In biological systems engineering, unlike the heavy workload of generating millions of biological sequences from a library for a protein or enzyme, contact matrix, which shows the physical touching within and between blocks in the amino acid sequence, is easily accessible. 
However, It is not crystal clear how to leverage the contact matrix to build a classifier for predicting the functionality of proteins from their amino acid sequence. 
In this paper, we present a new approach(simpler TAN) to incorporating the contact matrix based on Bayesian network framework. Theoretical study shows that our new algorithm has substantially improved computational efficiency relative to traditional TAN. In extensive experiments this technique delivers comparable or better prediction performance to NB and TAN in terms of different evaluation metrics. To further improve the classification performance and its stability without sacrificing computational efficiency, We propose another algorithm called Ensemble TAN. Finally, SHAP values are calculated to obtain feature importance scores to help us better understand the model. 

*Keywords:* Contact matrix, Bayesian network, Protein Sequence Analysis, Classifcation, SHAP
  
## Introduction
#### Motivation
Proteins perform a vast array of functions within organisms and they differ from one another primarily in their sequence of amino acids. Understanding the relationship between functionality of protein and their amino acid sequence has been of great interest to researchers. 
One particularly popular research is to apply Machine-learning approaches predicting how sequence maps to function in a data-driven manner without requiring a detailed model of the underlying physics or biological pathways.

Usually generating millions of biological sequences from a library for a protein or enzyme requires heavy workload, because proteins are large biomolecules, or macromolecules, consisting of one or more long chains of amino acid residues. 
In contrast, contact matrix, which shows the physical touching within and between blocks in the amino acid sequence, is easily accessible. It would be preferred if we can improve machine learning model by incorporating this cheap side information.
However, to the best of our knowledge, there is currently no published method available to leverage the contact matrix information to build a classifier for predicting the functionality of proteins from their amino acid sequence, and it is not cystal clear whether the efforts needed are worthwhile. 

As a concrete example which motivates this work, consider predicting the functionality of P450 and Lactamase from their amino acid sequence. 
In light of protein's three-dimensional arrangement of atoms in an amino acid-chain molecule, different positions in the amino acid sequence can be physically touching. To make the analysis simple, each sequence is divided into 8 blocks based on domain knowledge.
The way the contact matrix is calculated is to add up all the positions within and between blocks that are physically touching. Therefore, the contact matrix would be a symmetric eight by eight matrix. 
Typically Researchers would ignore the contact matrix regardless of its accessibility and build classifiers directly from amino acid sequence data using algorithms such as Logistic Regression, SVM, Naive Bayes and Neural network. However, Side information does provide useful knowledge in terms of understanding the closeness and interdependence among attributes, and this structure information may potentially imporve the classifier if used properly. 

To rigorously investigate whether a classifier delievers better prediction performance after incorporating the side information, we need to first clarify evaluation metrics to be used. 

#### Evaluation Metrics
Classification is one of the most important tasks in data mining, the predictive ability of a classifier is typically measured by its classification accuracy or error rate on the testing instances. However, evaluation of a classifier based purely on accuracy may suffer from the "Accuracy Paradox", For example, if the incidence of category A is dominant, being found in 99% of cases, then predicting that every case is category A will have an accuracy of 99%. To have a thorough and systematic investigation, Precision and recall should also be considered.

In fact, probability-based classifiers can also produce probability estimates or "confidence" of the class prediction. Unfortunately, this information is often ignored in classification. To further investigate the classification performance in terms of its class probability estimation, rather than just using the predicted classes information. Recently, conditional log likelihood, or simply CLL, has been used for this purpose and received a considerable attention.

Given a classifier G and a set of test instances T = {e_1,e_2,....e_t}, where t is the number of test instances. Let c_i be the true class label of e_i. Then the conditional log likelihood CLL(G|T) of the classifier G on the test set T is defined as:

CLL(G|T) = \sum_i^t log(P_{G} (c_i|e_i))

Let e represented by an attribute vector <a_1, a_2,....a_m> be a test instance and the true class label of it be c, then we can use the built classifier G to estimate the probability that e belongs to c. This resulting probability is generally called predicted probability denoted by \hat(G)(c|e). Now we can see that the classifiers with higher CLL tend to have better class probability estimation performance. In this work, we use all evaluation metrics mentioned to have a thorough understanding of classifier performance.


## Framework

#### Bayesian Network

#### Learning BN's

#### Naive Bayes

#### Tree Augmented Naive Bayes




## Simpler TAN + Ensemble TAN

#### Simpler TAN


#### Ensemble TAN

## Experiments and results


## Conclusions



## Acknowledgement 

## Reference

