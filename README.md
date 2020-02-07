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
On account of protein's three-dimensional arrangement of atoms in an amino acid-chain molecule, Different positions in the amino acid sequence can be physically touching. To make the analysis simple, each sequence is divided into 8 blocks based on domain knowledge.
The way the contact matrix is calculated is to add up all the positions within and between blocks that are physically touching. Therefore, the contact matrix would be a symmetric eight by eight matrix.
Typically Researchers would ignore the contact matrix and build classifiers directly from amino acid sequence data using algorithms such as Logistic Regression, SVM, Naive Bayes and Neural network. However, Side information does provide useful knowledge in terms of the closeness among attributes, and this structure information may potentially imporve the classifier if used properly. 



#### Prior Work and Our Contribution



## Review

#### Tree Augmented Naive Bayes




## Simpler TAN + Ensemble TAN

## Experiments and results


## Conclusions



## Acknowledgement 

## Reference

