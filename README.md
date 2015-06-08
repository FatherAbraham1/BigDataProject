##Support Vector Machine based Binary Hierarchical Decision Tree for multi-class classification Using Spark

####Abstract
Naïve Bayes and Logistic Regression were used for multi class classification. While Naïve Bayes will have excellent accuracy in predicting the classes for Binary classification, it performance dips when used in multi class classification. Logistic Regression is used generally for multi class classification and it has very high training cost.

Support Vector Machine has been proved to be a good tool in the field of Machine Learning, especially for classification. Since it was designed primarily for binary classification, one can employ various architectures to extend these for multi-class classification such as One-vs-One, One-vs-All and Directed Acyclic Graph (DAG).

I have employed Binary Hierarchical Decision Tree (BHDT), which uses K-Means clustering, since the time and space complexities are much better compared to the former approaches. The goal for this project is to efficiently learn a model of the dataset and develop a classifier, which is fast, lightweight and accurate. 
