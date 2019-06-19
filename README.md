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
Just like linear regression, Logistic regression is the right algorithm to start with classification algorithms. Eventhough, the name ‘Regression’ comes up, it is not a regression model, but a classification model. It uses a logistic function to frame binary output model. The output of the logistic regression will be a probability (0≤x≤1), and can be used to predict the binary 0 or 1 as the output ( if x<0.5, output= 0, else output=1).
#### Basic Theory :
Logistic Regression acts somewhat very similar to linear regression. It also calculates the linear output, followed by a stashing function over the regression output. Sigmoid function is the frequently used logistic function. You can see below clearly, that the z value is same as that of the linear regression output in Eqn(1).
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/logR1.PNG)

The h(θ) value here corresponds to P(y=1|x), ie, probability of output to be binary 1, given input x. P(y=0|x) will be equal to 1-h(θ).

when value of z is 0, g(z) will be 0.5. Whenever z is positive, h(θ) will be greater than 0.5 and output will be binary 1. Likewise, whenever z is negative, value of y will be 0. As we use a linear equation to find the classifier, the output model also will be a linear one, that means it splits the input dimension into two spaces with all points in one space corresponds to same label.

The figure below shows the distribution of a sigmoid function.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/logR2.PNG)

#### Loss function :
We can’t use mean squared error as loss function(like linear regression), because we use a non-linear sigmoid function at the end. MSE function may introduce local minimums and will affect the gradient descend algorithm.

So we use cross entropy as our loss function here. Two equations will be used, corresponding to y=1 and y=0. The basic logic here is that, whenever my prediction is badly wrong, (eg : y’ =1 & y = 0), cost will be -log(0) which is infinity.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/logR3.PNG)

In the equation given, m stands for training data size, y’ stands for predicted output and y stands for actual output.


#### Logistic regression can be used in cases such as:

- Predicting the Customer Churn
- Credit Scoring & Fraud Detection
- Measuring the effectiveness of marketing campaigns

#### Advantages :
- Easy, fast and simple classification method.
- θ parameters explains the direction and intensity of significance of independent variables over the dependent variable.
- Can be used for multiclass classifications also.
- Loss function is always convex.
#### Disadvantages :
- Cannot be applied on non-linear classification problems.
- Proper selection of features is required.
- Good signal to noise ratio is expected.
- Colinearity and outliers tampers the accuracy of LR model.
#### Hyperparameters :
Logistic regression hyperparameters are similar to that of linear regression. Learning rate(α) and Regularization parameter(λ) have to be tuned properly to achieve high accuracy.

#### Assumptions of LR :
Logistic regression assumptions are similar to that of linear regression model. please refer the above section.

#### Comparison with other models :
##### Logistic regression vs SVM :

- SVM can handle non-linear solutions whereas logistic regression can only handle linear solutions.
- Linear SVM handles outliers better, as it derives maximum margin solution.
- Hinge loss in SVM outperforms log loss in LR.
##### Logistic Regression vs Decision Tree :

- Decision tree handles colinearity better than LR.
- Decision trees cannot derive the significance of features, but LR can.
- Decision trees are better for categorical values than LR.
##### Logistic Regression vs Neural network :

- NN can support non-linear solutions where LR cannot.
- LR have convex loss function, so it wont hangs in a local minima, whereas NN may hang.
- LR outperforms NN when training data is less and features are large, whereas NN needs large training data.
##### Logistic Regression vs Naive Bayes :

- Naive bayes is a generative model whereas LR is a discriminative model.
- Naive bayes works well with small datasets, whereas LR+regularization can achieve similar performance.
- LR performs better than naive bayes upon colinearity, as naive bayes expects all features to be independent.
##### Logistic Regression vs KNN :

- KNN is a non-parametric model, where LR is a parametric model.
- KNN is comparatively slower than Logistic Regression.
- KNN supports non-linear solutions where LR supports only linear solutions.
- LR can derive confidence level (about its prediction), whereas KNN can only output the labels.

#### SOURCES 
```
https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222
https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f
```

### K-Nearest Neighbors (K-NN)
K-nearest neighbors is a non-parametric method used for classification and regression. It is one of the most easy ML technique used. It is a lazy learning model, with local approximation.

#### Basic Theory :
The basic logic behind KNN is to explore your neighborhood, assume the test datapoint to be similar to them and derive the output. In KNN, we look for k neighbors and come up with the prediction.

In case of KNN classification, a majority voting is applied over the k nearest datapoints whereas, in KNN regression, mean of k nearest datapoints is calculated as the output. As a rule of thumb, we selects odd numbers as k. KNN is a lazy learning model where the computations happens only runtime.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/knn1.PNG)

In the above diagram yellow and violet points corresponds to Class A and Class B in training data. The red star, points to the testdata which is to be classified. when k = 3, we predict Class B as the output and when K=6, we predict Class A as the output.

#### Loss function :
There is no training involved in KNN. During testing, k neighbors with minimum distance, will take part in classification /regression.

#### Advantages :
Easy and simple machine learning model.
Few hyperparameters to tune.
#### Disadvantages :
k should be wisely selected.
Large computation cost during runtime if sample size is large.
Proper scaling should be provided for fair treatment among features.
#### Hyperparameters :
KNN mainly involves two hyperparameters, K value & distance function.

#### K value : how many neighbors to participate in the KNN algorithm. k should be tuned based on the validation error.
distance function : Euclidean distance is the most used similarity function. Manhattan distance, Hamming Distance, Minkowski distance are different alternatives.
#### Assumptions :
There should be clear understanding about the input domain.
feasibly moderate sample size (due to space and time constraints).
colinearity and outliers should be treated prior to training.
#### Comparison with other models :
A general difference between KNN and other models is the large real time computation needed by KNN compared to others.

#### KNN vs naive bayes :

Naive bayes is much faster than KNN due to KNN’s real-time execution.
Naive bayes is parametric whereas KNN is non-parametric.
#### KNN vs linear regression :

KNN is better than linear regression when the data have high SNR.
#### KNN vs SVM :

SVM take cares of outliers better than KNN.
If training data is much larger than no. of features(m>>n), KNN is better than SVM. SVM outperforms KNN when there are large features and lesser training data.
#### KNN vs Neural networks :

Neural networks need large training data compared to KNN to achieve sufficient accuracy.
NN needs lot of hyperparameter tuning compared to KNN.

```
https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222
https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f
https://www.fromthegenesis.com/pros-and-cons-of-k-nearest-neighbors/
```

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