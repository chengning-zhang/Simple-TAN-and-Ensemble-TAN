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
The way the contact matrix is calculated is to add up all the positions within and between blocks that are physically touching. Therefore, the contact matrix would be a symmetric 8 by 8 matrix. 
Typically Researchers would ignore the contact matrix regardless of its accessibility and build classifiers directly from amino acid sequence data using algorithms such as Logistic Regression, SVM, Naive Bayes and Neural network. However, Side information does provide useful knowledge in terms of understanding the closeness and interdependence among attributes, and this structure information may potentially imporve the classifier if used properly. 

To rigorously investigate whether a classifier delievers better prediction performance after incorporating the side information, we need to first clarify evaluation metrics to be used. 

#### Evaluation Metrics
Classification is one of the most important tasks in data mining, the predictive ability of a classifier is typically measured by its classification accuracy or error rate on the testing instances. However, evaluation of a classifier based purely on accuracy may suffer from the "Accuracy Paradox", For example, if the incidence of category A is dominant, being found in 99% of cases, then predicting that every case is category A will have an accuracy of 99%. To have a thorough and systematic investigation, Precision and recall should also be considered.

In fact, probability-based classifiers can also produce probability estimates or "confidence" of the class prediction. Unfortunately, this information is often ignored in classification. A nature question is how to evaluate the classification performance in terms of its class probability estimation, rather than just using the predicted classes information. Recently, conditional log likelihood, or simply CLL, has been used for this purpose and received a considerable attention.
Given a classifier G and a set of test instances T = {e_1,e_2,....e_t}, where t is the number of test instances. Let c_i be the true class label of e_i. Then the conditional log likelihood CLL(G|T) of the classifier G on the test set T is defined as:

CLL(G|T) = \sum_i^t log(P_{G} (c_i|e_i))

Let e represented by an attribute vector <a_1, a_2,....a_m> be a test instance and the true class label of it be c, then we can use the built classifier G to estimate the probability that e belongs to c. This resulting probability is generally called predicted probability denoted by \hat(G)(c|e). Now we can see that the classifiers with higher CLL tend to have better class probability estimation performance. In this work, we use all evaluation metrics mentioned to have a thorough understanding of classifier performance.

The rest of the paper is organized as follows. At first, we briefly review Bayesian Network framework upon which our new method to incorporate side information is based. Then, we present our new algorithms respectively Simpler TAN and Ensemble TAN and provide theoretical time efficiency guarantee. Followed by the description of our experiments and results in detail. and SHAP is applied to obtain attributes importance scores. Lastly we draw conclusions.

## Framework

#### Bayesian Network
A Bayesian network B = <N,A, \theta> is a directed acyclic graph (DAG) <N,A> with a conditional probability distribution (CP table) for each node, collectively represented by \theta. Each node n\in N represents an attibute, and each arc a\in A between nodes represents a probabilistic dependency. In general, a BN can be used to compute the conditional probability of one node, given values assigned to the other nodes; hence a BN can be used as a classifier that gives the posterial probability distrbution of the classification node given the values of other attributes.  

In summary, BN can be viewed as a "data structure that provides the skeleton for representing a joint distribution compactly in a factorized way and a compact representation for a set of conditional independent assumptions about a distribution."(cited). In order to build a classifcation model, we need to determine the joint distribution of all the attributes conditioned on class. The formula is given by ,

P(C|a_1,...a_m) = \frac{P(C). P(a_1,...a_m|C)}{P(a_1,...a_m)}

The general problem of computing the joint posterial probabilities in BN is NP-hard(Cooper 1990). To reduce the complexity,
Some restrictions need to be imposed on the level of interaction between attributes.

#### Learning BN's
The two major tasks in learning a BN are: learning the graphic structure; and then learning the parameters(CP table entries) for that structure. As it is trivial to learn the parameters for a given structure that are optimal for a given corpus of complete data -- simply use the emopirical conditional frequencies from the data(cited). We will focus on learning the BN structure. 

There are two approaches to learning BN structure. First one is the scoring-based learning algorithms, that seek a structure that maximize the Bayesian, MDL or Kullback-Leibler(KL) entropy scoring function(cited).
Second approach is to find the conditional independence relationships among the attributes and use these relationships as constraints to construct a BN. These algorithms are referred as CI-based algorithms(cited).

Heckerman et al(1997) compare these two general learning and show that the scoring-based methods often have certain advantages over the CI-based methods, in terms of modeling a distribution. However, Friedman et al(1997) show theoretically that the scoring-based methods may result in poor classifiers since a good classifier maximize a different function. In summary, the scoring-based methods are often less useful in practice.

#### Naive Bayes
A Naive Bayes BN, is a simple structure that has the classification node as the parent node of all other nodes. No other connections are allowed in a Naive-Bayes structure.
Naive Bayes has been used as an effective classifier for many years. 
NB estimate class probability using:

P(C|a_1,...a_m) = \frac{P(C). \prod_1^m P(a_j|C) } { P(a_1,...a_m)}

Naive Bayes is the simplest form of Bayesian network classifiers. It is obvious that the conditional independence assumption in Naive Bayes is rarely true in reality, which would harm its performance in the applications with complex attribute dependencies. Numerous algorithms have been proposed to improve Naive nayes by weakening its conditional attribute independence assumption, among which Tree augmented Naive Bayes TAN has demonstrated remarkable classification performance and is competitive with the general Bayesian network classifiers in terms of accuracy, while maintaining efficiency and simplicity.

#### Tree Augmented Naive Bayes
In order to weaken the conditional independence assumption of Naive Bayes effectively, and an appropriate language and efficient machinery to represent the independence assertions are needed(cited). Unfortunately, it has been proved that learning an optimal Bayesian network is NP-hard(cited). In order to avoid the intractable complexity for learning Bayesian networks, we need to impose restrictions on structure of Bayesian network.

TAN is an extented tree-like Naive Bayes, in which the class node directly points to all attributes nodes and an attribute node only has at most one parent from another attribute node. TAN is a specific case of general Bayesian network classifiers, in which the class node also directly points to all attribute nodes(except that they do not form any directed cycle).
Assume that A_1,A_2,...A_m are m attributes and C is the class variable, the learning algorithm of TAN is depicted as:





## Simpler TAN + Ensemble TAN

#### Simpler TAN


#### Ensemble TAN

## Experiments and results


## Conclusions



## Acknowledgement 

## Reference

