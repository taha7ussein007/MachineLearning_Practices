# Machine Learning Algorithms Basic Information
This information from general sources mainly i'm studying from Udemy - Machine Learning from A to Z

## Regression
### Simple Linear Regression & Multiple Linear Regression
If you want to start machine learning, Linear regression is the best place to start.
Linear Regression is a regression model, meaning, it’ll take features and predict a [continuous] output, eg : stock price,salary etc. Linear regression as the name says, finds a linear curve solution to every problem.
#### Basic Theory :
LR allocates weight parameter, theta for each of the training features. 
The predicted output(h(θ)) will be a linear function of features and θ coefficients.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/lr1.png)
During the start of training, each theta is randomly initialized. But during the training, we correct the theta corresponding to each feature such that, the loss (metric of the deviation between expected and predicted output) is minimized. Gradient descend algorithm will be used to align the θ values in the right direction. In the below diagram, each red dots represent the training data and the blue line shows the derived solution.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/lr2.gif)

#### Loss function :
- In LR, we use mean squared error as the metric of loss. 
- The deviation of expected and actual outputs will be squared and sum up. 
- Derivative of this loss will be used by gradient descend algorithm.

#### Advantages :
- Easy and simple implementation.
- Space complex solution.
- Fast training.
- Value of θ coefficients gives an assumption of feature significance.
#### Disadvantages :
- Applicable only if the solution is linear. 
- In many real life scenarios, it may not be the case.
- Algorithm assumes the input residuals (error) to be normal distributed, but may not be satisfied always.
- Algorithm assumes input features to be mutually independent(no co-linearity).
#### Hyperparameters :
- Regularization parameter (λ) : Regularization is used to avoid over-fitting on the data. 
- Higher the λ, higher will be regularization and the solution will be highly biased. 
- Lower the λ, solution will be of high variance. 
- An intermediate value is preferable.
- learning rate (α) : it estimates, by how much the θ values should be corrected while applying gradient descend algorithm during training. 
- α should also be a moderate value.
#### Assumptions for LR :
- Linear relationship between the independent and dependent variables.
- Training data to be homoskedastic, meaning the variance of the errors should be somewhat constant.
- Independent variables should not be co-linear.
- Indepence of errors
#### Colinearity & Outliers :
- Two features are said to be colinear when one feature can be linearly predicted from the other with somewhat accuracy.
- colinearity will simply inflate the standard error and causes some significant features to become insignificant during training. 
- Ideally, we should calculate the colinearity prior to training and keep only one feature from highly correlated feature sets.
- Outlier is another challenge faced during training. 
- They are data-points that are extreme to normal observations and affects the accuracy of the model.
- outliers inflates the error functions and affects the curve function and accuracy of the linear regression. 
- Regularization (especially L1 ) can correct the outliers, by not allowing the θ parameters to change violently.
- During Exploratory data analysis phase itself, we should take care of outliers and correct/eliminate them.
- Box-plot can be used for identifying them.
#### Comparison with other models :
- As the linear regression is a regression algorithm, we will compare it with other regression algorithms. 
- One basic difference of linear regression is, LR can only support linear solutions. 
- There are no best models in machine learning that outperforms all others(no free Lunch), and efficiency is based on the type of training data distribution.

##### LR vs Decision Tree :
- Decision trees supports non linearity, where LR supports only linear solutions.
- When there are large number of features with less data-sets(with low noise), linear regressions may outperform Decision trees/random forests. In general cases, Decision trees will be having better average accuracy.
- For categorical independent variables, decision trees are better than linear regression.
- Decision trees handles colinearity better than LR.

##### LR vs SVM :

- SVM supports both linear and non-linear solutions using kernel trick.
- SVM handles outliers better than LR.
- Both perform well when the training data is less, and there are large number of features.

##### LR vs KNN :

- KNN is a non -parametric model, whereas LR is a parametric model.
- KNN is slow in real time as it have to keep track of all training data and find the neighbor nodes, whereas LR can easily extract output from the tuned θ coefficients.

##### LR vs Neural Networks :

- Neural networks need large training data compared to LR model, whereas LR can work well even with less training data.
- NN will be slow compared to LR.
- Average accuracy will be always better with neural networks.

#### SOURCES 
```
https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222
https://medium.com/@kabab/linear-regression-with-python-d4e10887ca43
http://r-statistics.co/Assumptions-of-Linear-Regression.html
```

### Polynomial Regression
In the last section, we saw two variables in your data set were correlated but what happens if we know that our data is correlated, but the relationship doesn’t look linear? So hence depending on what the data looks like, we can do a polynomial regression on the data to fit a polynomial equation to it.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr1.gif)
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr2.gif)

Hence If we try to use a simple linear regression in the above graph then the linear regression line won’t fit very well. It is very difficult to fit a linear regression line in the above graph with a low value of error. Hence we can try to use the polynomial regression to fit a polynomial line so that we can achieve a minimum error or minimum cost function.
This is the general equation of a polynomial regression is:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr3.png)

##### Advantages of using Polynomial Regression:
- Polynomial provides the best approximation of the relationship between the dependent and independent variable.
- A Broad range of function can be fit under it.
- Polynomial basically fits a wide range of curvature.

##### Disadvantages of using Polynomial Regression
- The presence of one or two outliers in the data can seriously affect the results of the nonlinear analysis.
- These are too sensitive to the outliers.
- In addition, there are unfortunately fewer model validation tools for the detection of outliers in nonlinear regression than there are for linear regression.

#### SOURCES 
```
https://towardsdatascience.com/introduction-to-linear-regression-and-polynomial-regression-f8adc96f31cb
```


### Support Vector Regression
### Decission Tree Regression
### Random Forest Regression

## Classification
### Logistic Regression
### K-Nearest Neighbors (K-NN)
### Support Vector Machine (SVM)
### Kernel SVM
### Naive Bayes
### Decision Tree Classification
### Random Forest Classification

## Clustering
### K-Means Clustering
### Hierarchical Clustering

## Association Rule Learning
### Apriori
### Eclat

## Natural Language Processing (English) 


## Reinforcement Learning
### Upper Confidence Bound (UCB)
### Thompson Sampling

## Dimensionality Reduction 
### Principal Component Analysis (PCA)
### Linear Discriminant Analysis (LDA)
### Kernel PCA

## Model Selection and Boosting
### Model Selection	
### XGBoost