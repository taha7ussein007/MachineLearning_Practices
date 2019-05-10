# Machine Learning Algorithms Basic Information
This information from general sources mainly i'm studying from Udemy - Machine Learning from A to Z

## Regression
### Simple Linear Regression & Multiple Linear Regression
If you want to start machine learning, Linear regression is the best place to start.
Linear Regression is a regression model, meaning, it’ll take features and predict a [continuous] output, eg : stock price,salary etc. Linear regression as the name says, finds a linear curve solution to every problem.
#### Basic Theory :
LR allocates weight parameter, theta for each of the training features. 
The predicted output(h(θ)) will be a linear function of features and θ coefficients.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/lr1.PNG)
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
https://www.listendata.com/2014/11/difference-between-linear-regression.html
```

### Polynomial Regression
In the last section, we saw two variables in your data set were correlated but what happens if we know that our data is correlated, but the relationship doesn’t look linear? So hence depending on what the data looks like, we can do a polynomial regression on the data to fit a polynomial equation to it.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr1.gif)
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr2.gif)

Hence If we try to use a simple linear regression in the above graph then the linear regression line won’t fit very well. It is very difficult to fit a linear regression line in the above graph with a low value of error. Hence we can try to use the polynomial regression to fit a polynomial line so that we can achieve a minimum error or minimum cost function.
This is the general equation of a polynomial regression is:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/plr3.PNG)

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
This post is about SUPPORT VECTOR REGRESSION. Those who are in Machine Learning or Data Science are quite familiar with the term SVM or Support Vector Machine. But SVR is a bit different from SVM. As the name suggest the SVR is an regression algorithm , so we can use SVR for working with continuous Values instead of Classification which is SVM.

The terms that we are going to be using frequently in this post

1- Kernel: The function used to map a lower dimensional data into a higher dimensional data.
2- Hyper Plane: In SVM this is basically the separation line between the data classes. Although in SVR we are going to define it as the line that will will help us predict the continuous value or target value
3- Boundary line: In SVM there are two lines other than Hyper Plane which creates a margin . The support vectors can be on the Boundary lines or outside it. This boundary line separates the two classes. In SVR the concept is same.
4- Support vectors: This are the data points which are closest to the boundary. The distance of the points is minimum or least.
#### Why SVR ? Whats the main difference between SVR and a simple regression model?
In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold. This might be a bit confusing but let me explain.

#### Simply 
What we are trying to do here is basically trying to decide a decision boundary at ‘e’ distance from the original hyper plane such that data points closest to the hyper plane or the support vectors are within that boundary line
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svr1.PNG)


#### SOURCES 
```
https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff
https://www.researchgate.net/figure/Schematic-of-the-one-dimensional-support-vector-regression-SVR-model-Only-the-points_fig5_320916953
```

### Decission Tree Regression
Simply it is a non linear and non continuous regressor and it try to take average for each range of values 
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART1.PNG)
CART stand for classification and regression tree:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART2.PNG)

Regression are a bit more complex than classsification and require more attention, so i will try to break this complex topic in a simple way.

So here we got a scatter plot which represents data here we got 2 independent variables, and what we are predicting is a 3rd variable Y. You can’t see it into this model because its a bidimensional chart, but imagine Y as the 3rd dimension sticking out of your screen.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART3.PNG)

So if we add a 3rd dimesion it would look like at in the top right window, but we don’t need to see Y because we first need to work with this scatterplot and see how decision tree is created. So once we run the decision tree algorithm in the regression sense of it what would happen is that your scaterplot would be split up in segments. So lets have a look how this algorithm would do it:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART4.PNG)

Now how and where these splits are conducted is determined by the algorithm which uses mathematical entrophy when performing the split. This split increases the amount of information on the scattered points by adding some values and ultimately stop when it cannot add anymore information to our setup and splits these segments in leaves. For example when we have less than 5% of our total points in our local leaf, that segment wouldn’t be created, so the most important thing is where the split happens. And to in order to have a more in-depth understanding of how the split is determined you need to dive into mathematical entrophy, but this will not be covered here as it is beyond the scope of this article.

For now is sufficient to know that the algorithm will handle this, and find the optimal number of leafs which will be call terminal leafs.

So lets rewind all of it once again and create split one. As we can see the first split Split 1 happens at 20, so here we basically have two options. Then Split 2 happens at 170 so that means that condition x1<20 is satisfied. We will go ahead and create Split 3 at 200 and Split 4 at 40:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART5.PNG)

So when our decision tree is is drawn we populate the boxes, where we consider our dependent variable and check for paths of new obervation which are added to our dataset. So lets say we add an observation x1=30 and x2= 50, it would fall between terminal leafs 1 and 3 and this information helps us in predicting the value of Y. The way this works is pretty straight forward: we just take the average of each of theterminal leafs so we calculate the average of all those points, and that value will be assigned to the entire terminal leaf.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/CART6.PNG)

#### Advantages of CART
- Simple to understand, interpret, visualize.
- Decision trees implicitly perform variable screening or feature selection.
- Can handle both numerical and categorical data. Can also handle multi-output problems.
- Decision trees require relatively little effort from users for data preparation.
- Nonlinear relationships between parameters do not affect tree performance.
#### Disadvantages of CART
- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This is called variance, which needs to be lowered by methods like bagging and boosting.
- Greedy algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees, where the features and samples are randomly sampled with replacement.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the data set prior to fitting with the decision tree.

#### Tips and Tricks
Can help you decide on whether or not it’s the right model for your problem :-

##### Pros
- Easy to understand and interpret. At each node, we are able to see exactly what decision our model is making. In practice we’ll be able to fully understand where our accuracies and errors are coming from, what type of data the model would do well with, and how the output is influenced by the values of the features. Scikit learn’s visualisation tool is a fantastic option for visualising and understanding decision trees.
- Require very little data preparation. Many ML models may require heavy data pre-processing such as normalization and may require complex regularisation schemes. Decision trees on the other hand work quite well out of the box after tweaking a few of the parameters.
- The cost of using the tree for inference is logarithmic in the number of data points used to train the tree. That’s a huge plus since it means that having more data won’t necessarily make a huge dent in our inference speed.

##### Cons
- Overfitting is quite common with decision trees simply due to the nature of their training. It’s often recommended to perform some type of dimensionality reduction such as PCA so that the tree doesn’t have to learn splits on so many features
- For similar reasons as the case of overfitting, decision trees are also vulnerable to becoming biased to the classes that have a majority in the dataset. It’s always a good idea to do some kind of class balancing such as class weights, sampling, or a specialised loss function.

#### SOURCES 
```
https://medium.com/data-py-blog/decision-tree-regression-in-python-b185a3c63f2b
https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052
https://towardsdatascience.com/a-guide-to-decision-trees-for-machine-learning-and-data-science-fe2607241956
```

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