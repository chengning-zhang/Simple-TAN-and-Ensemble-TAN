# Simpler TAN + Ensemble TAN

A custom implementation of Bayesian network written from scratch in Python 3, API inspired by SciKit-learn. 
This module implements 4 Bayesian classifiers: 
* Naive Bayes
* TAN
* Simpler TAN
* Ensemble TAN


## Getting Started

This module is dependent on scikit-learn library. Make sure scikit-learn is properly installed.

### Prerequisites

Install sklearn, SHAP, pyitlib

```
pip install sklearn
pip install SHAP
pip install pyitlib
```

### Installing


```
Give the example
```

And repeat

```
until finished
```

## Using classifiers

Toy example on Naive Bayes

```
import numpy as np
Y = np.array([1, 1, 1, 2, 2, 2])
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
import NB
nb = NB(alpha = 1)
nb.fit(X,y)
nb.get_params()
nb.classes_
print(nb.name)
print(nb.predict_proba(X))
nb.score(X,y)
```

Toy example on Ensemble TAN

```
import numpy as np
Y = np.array([1, 1, 1, 2, 2, 2])
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
import STAN_TAN_bagging
stan_tan_bag = STAN_TAN_bagging(alpha = 1)
stan_tan_bag.fit(X,y,M)
stan_tan_bag.get_params()
stan_tan_bag.classes_
print(stan_tan_bag.name)
print(stan_tan_bag.predict_proba(X))
stan_tan_bag.score(X,y)
```

## Get cross validation


```
import get_cv

Y = np.array([1, 1, 1, 2, 2, 2])
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Accuracy, CLL, training_time,Precision,Recall= get_cv(NB,X,y,M = None)
print(np.mean(Accuracy))
print(np.mean(CLL))
print(np.mean(Precision))
print(np.mean(Recall))
print(np.mean(np.array(training_time)))

```

## Model explanation using SHAP

```
nb = NB()
nb.fit(lactamase)
explainer1 = shap.KernelExplainer(nb.predict_binary, X2[0:50,], link="logit")
shap_values1 = explainer1.shap_values(X2,nsamples = 20)

shap.summary_plot(shap_values1, X2)
shap.summary_plot(shap_values1, X2, plot_type="bar")

```




## Built With

* [Dropwizard](https://scikit-learn.org/stable/modules/classes.html) - scikit-learn API
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Chengning Zhang** - *Initial work* - [PurpleBooth](https://github.com/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
