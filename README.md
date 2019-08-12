# Machine Learning Algorithms Basic Information
This information from general sources but mainly i'm studying from Udemy - Machine Learning from A to Z
i'm trying to collect special notes and explaintions in case the articles were lost or removed 

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

### Support Vector Machine (SVM) + Kernel SVM (Non Linear)

##### 1. What is SVM?
Support vector machines so called as SVM is a supervised learning algorithm which can be used for classification and regression problems as support vector classification (SVC) and support vector regression (SVR). It is used for smaller dataset as it takes too long to process. In this set, we will be focusing on SVC.
##### 2. The ideology behind SVM:
SVM is based on the idea of finding a hyperplane that best separates the features into different domains.
##### 3. Intuition development:
Consider a situation following situation:
There is a stalker who is sending you emails and now you want to design a function( hyperplane ) which will clearly differentiate the two cases, such that whenever you received an email from the stalker it will be classified as a spam. The following are the figure of two cases in which the hyperplane are drawn, which one will you pick and why? take a moment to analyze the situation ……
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm1.PNG)

I guess you would have picked the fig(a). Did you think why have you picked the fig(a)? Because the emails in fig(a) are clearly classified and you are more confident about that as compared to fig(b). Basically, SVM is composed of the idea of coming up with an Optimal hyperplane which will clearly classify the different classes(in this case they are binary classes).
##### 4. Terminologies used in SVM:
The points closest to the hyperplane are called as the support vector points and the distance of the vectors from the hyperplane are called the margins.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm2.PNG)

The basic intuition to develop over here is that more the farther SV points, from the hyperplane, more is the probability of correctly classifying the points in their respective region or classes. SV points are very critical in determining the hyperplane because if the position of the vectors changes the hyperplane’s position is altered. Technically this hyperplane can also be called as margin maximizing hyperplane.
##### 5. Hyperplane(Decision surface ):
For so long in this post we have been discussing the hyperplane, let’s justify its meaning before moving forward. The hyperplane is a function which is used to differentiate between features. In 2-D, the function used to classify between features is a line whereas, the function used to classify the features in a 3-D is called as a plane similarly the function which classifies the point in higher dimension is called as a hyperplane. Now since you know about the hyperplane lets move back to SVM.
Let’s say there are “m” dimensions:
thus the equation of the hyperplane in the ‘M’ dimension can be given as =

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm3.PNG)

##### 6. Hard margin SVM:
Now,
Assume 3 hyperplanes namely (π, π+, π−) such that ‘π+’ is parallel to ‘π’ passing through the support vectors on the positive side and ‘π−’ is parallel to ‘π’ passing through the support vectors on the negative side.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm4.PNG)

the equations of each hyperplane can be considered as:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm5.PNG)

for the point X1 :

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm6.PNG)

Explanation: when the point X1 we can say that point lies on the hyperplane and the equation determines that the product of our actual output and the hyperplane equation is 1 which means the point is correctly classified in the positive domain.
for the point X3:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm7.PNG)

Explanation: when the point X3 we can say that point lies away from the hyperplane and the equation determines that the product of our actual output and the hyperplane equation is greater 1 which means the point is correctly classified in the positive domain.
for the point X4:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm8.PNG)

Explanation: when the point X4 we can say that point lies on the hyperplane in the negative region and the equation determines that the product of our actual output and the hyperplane equation is equal to 1 which means the point is correctly classified in the negative domain.
for the point X6 :

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm9.PNG)

Explanation: when the point X6 we can say that point lies away from the hyperplane in the negative region and the equation determines that the product of our actual output and the hyperplane equation is greater 1 which means the point is correctly classified in the negative domain.
Let’s look into the constraints which are not classified:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm10.PNG)

for point X7:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm11.PNG)


Explanation: When Xi = 7 the point is classified incorrectly because for point 7 the wT + b will be smaller than one and this violates the constraints. So we found the misclassification because of constraint violation. Similarly, we can also say for points Xi = 8.
Thus from the above examples, we can conclude that for any point Xi,
if Yi(WT*Xi +b) ≥ 1:
then Xi is correctly classified
else:
Xi is incorrectly classified.
So we can see that if the points are linearly separable then only our hyperplane is able to distinguish between them and if any outlier is introduced then it is not able to separate them. So these type of SVM is called as hard margin SVM (since we have very strict constraints to correctly classify each and every datapoint).
##### 7. Soft margin SVM:
We basically consider that the data is linearly separable and this might not be the case in real life scenario. We need an update so that our function may skip few outliers and be able to classify almost linearly separable points. For this reason, we introduce a new Slack variable ( ξ ) which is called Xi.
if we introduce ξ it into our previous equation we can rewrite it as

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm12.PNG)

Introduction of Xi
if ξi= 0,
the points can be considered as correctly classified.
else:
ξi> 0 , Incorrectly classified points.
so if ξi> 0 it means that Xi(variables)lies in incorrect dimension, thus we can think of ξi as an error term associated with Xi(variable). The average error can be given as;

average error

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm13.PNG)

thus our objective, mathematically can be described as;
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm14.PNG)



where ξi = ςi
READING: To find the vector w and the scalar b such that the hyperplane represented by w and b maximizes the margin distance and minimizes the loss term subjected to the condition that all points are correctly classified.
This formulation is called the Soft margin technique.
##### 8. Loss Function Interpretation of SVM:
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm15.PNG)


thus it can be interpreted that hinge loss is max(0,1-Zi).
##### 9. Dual form of SVM:
Now, let’s consider the case when our data set is not at all linearly separable.
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm16.PNG)

basically, we can separate each data point by projecting it into the higher dimension by adding relevant features to it as we do in logistic regression. But with SVM there is a powerful way to achieve this task of projecting the data into a higher dimension. The above-discussed formulation was the primal form of SVM . The alternative method is dual form of SVM which uses Lagrange’s multiplier to solve the constraints optimization problem.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm17.PNG)

Note:
If αi>0 then Xi is a Support vector and when αi=0 then Xi is not a support vector.
Observation:
To solve the actual problem we do not require the actual data point instead only the dot product between every pair of a vector may suffice.
To calculate the “b” biased constant we only require dot product.
The major advantage of dual form of SVM over Lagrange formulation is that it only depends on the α.
##### 10. What is Kernel trick?
Coming to the major part of the SVM for which it is most famous, the kernel trick. The kernel is a way of computing the dot product of two vectors x and y in some (very high dimensional) feature space, which is why kernel functions are sometimes called “generalized dot product.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm18.PNG)

try reading this equation…

Applying kernel trick means just to the replace dot product of two vectors by the kernel function.
##### 11. Types of kernels:
linear kernel
polynomial kernel
Radial basis function kernel (RBF)/ Gaussian Kernel
We will be focusing on the polynomial and Gaussian kernel since its most commonly used.
Polynomial kernel:
In general, the polynomial kernel is defined as ;

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm19.PNG)

b = degree of kernel & a = constant term.
in the polynomial kernel, we simply calculate the dot product by increasing the power of the kernel.
Example:
Let’s say originally X space is 2-dimensional such that
Xa = (a1 ,a2)
Xb = (b1 ,b2)
now if we want to map our data into higher dimension let’s say in Z space which is six-dimensional it may seem like

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm20.PNG)

In order to solve the solve this dual SVM we would require the dot product of (transpose) Za ^t and Zb.
- Method 1:
traditionally we would solve this by :

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm21.PNG)

which will a lot of time as we would have to performs dot product on each datapoint and then to compute the dot product we may need to do multiplications Imagine doing this for thousand datapoints….
Or else we could simply use
- Method 2:
using kernel trick:

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm22.PNG)

In this method, we can simply calculate the dot product by increasing the value of power. Simple isn’t it?
Radial basis function kernel (RBF)/ Gaussian Kernel:
Gaussian RBF(Radial Basis Function) is another popular Kernel method used in SVM models for more. RBF kernel is a function whose value depends on the distance from the origin or from some point. Gaussian Kernel is of the following format;

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm23.PNG)

Using the distance in the original space we calculate the dot product (similarity) of X1 & X2.
Note: similarity is the angular distance between two points.
Parameters:
C: Inverse of the strength of regularization.
Behavior: As the value of ‘c’ increases the model gets overfits.
As the value of ‘c’ decreases the model underfits.
2. γ : Gamma (used only for RBF kernel)
Behavior: As the value of ‘ γ’ increases the model gets overfits.
As the value of ‘ γ’ decreases the model underfits.

* Another simple point of view for gaussian kernel
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm-0.PNG)
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm-1.PNG)
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm-2.PNG)

* Summary for kernels 
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/svm-3.PNG)

#####  12. Pros and cons of SVM:
Pros:
It is really effective in the higher dimension.
Effective when the number of features are more than training examples.
Best algorithm when classes are separable
The hyperplane is affected by only the support vectors thus outliers have less impact.
SVM is suited for extreme case binary classification.
cons:
For larger dataset, it requires a large amount of time to process.
Does not perform well in case of overlapped classes.
Selecting, appropriately hyperparameters of the SVM that will allow for sufficient generalization performance.
Selecting the appropriate kernel function can be tricky.
#####  13. Preparing data for SVM:
1. Numerical Conversion:
SVM assumes that you have inputs are numerical instead of categorical. So you can convert them using one of the most commonly used “one hot encoding , label-encoding etc”.
2. Binary Conversion:
Since SVM is able to classify only binary data so you would need to convert the multi-dimensional dataset into binary form using (one vs the rest method / one vs one method) conversion method.

#### SOURCES 
```
https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589
http://cs229.stanford.edu/notes/cs229-notes3.pdf
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
```

### Naive Bayes
```
This algorithm is called “Naive” because it makes a naive assumption that each feature is independent of other features which is not true in real life.
```
As for the “Bayes” part, it refers to the statistician and philosopher, Thomas Bayes and the theorem named after him, Bayes’ theorem, which is the base for Naive Bayes Algorithm.

##### What is Naive Bayes Algorithm?
On Summarizing the above mentioned points Naive Bayes algorithm can be defined as a supervised classification algorithm which is based on Bayes theorem with an assumption of independence among features.

##### A Brief look on Bayes Theorem :
Bayes Theorem helps us to find the probability of a hypothesis given our prior knowledge.
As per wikipedia,In probability theory and statistics, Bayes’ theorem (alternatively Bayes’ law or Bayes’ rule, also written as Bayes’s theorem) describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
Lets look at the equation for Bayes Theorem,
![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/nb1.PNG)

Where,
- P(A|B) is the probability of hypothesis A given the data B. This is called the posterior probability.
- P(B|A) is the probability of data B given that the hypothesis A was true.
- P(A) is the probability of hypothesis A being true (regardless of the data). This is called the prior probability of A.
- P(B) is the probability of the data (regardless of the hypothesis).
If you are thinking what is P(A|B) or P(B|A)?These are conditional probabilities having formula :

If you still have confusion,this image summarizes Bayes Theorem-

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/nb2.PNG)

##### How does Naive Bayes Algorithm work?
Let us take an example to understand how does Naive Bayes Algorithm work.
Suppose we have a training dataset of 1025 fruits.The feature in the dataset are these : Yellow_color,Big_Size,Sweet_Taste.There are three different classes apple,banana & others.
Step 1: Create a frequency table for all features against all classes

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/nb3.PNG)

##### What can we conclude from the above table?
- Out of 1025 fruits, 400 are apples, 525 are bananas, and 100 are others.
- 175 of the total 400 apples are Yellow and the rest are not and so on.
- 400 fruits are Yellow, 425 are big in size and 200 are sweet from a total of 600 fruits.
Step 2: Draw the likelihood table for the features against the classes.

![alt text](https://github.com/taha7ussein007/MachineLearning_Practices/blob/master/images/nb4.PNG)

In our likelihood table Total_Probability of banana is maximum(0.1544) when the fruit is of Yellow_Color,Big in size and Sweet in taste.Therefore as per Naive Bayes algorithm a fruit which is Yellow in color,big in size and sweet in taste is Banana.
In a nutshell, we say that a new element will belong to the class which will have the maximum conditional probability described above.
##### Pros and Cons of Naive Bayes Algorithm:
###### Pros :
1- It is easy to understand.
2- It can also be trained on small dataset.
###### Cons :
- It has a ‘Zero conditional probability Problem’, for features having zero frequency the total probability also becomes zero.There are several sample correction techniques to fix this problem such as “Laplacian Correction.”
- Another disadvantage is the very strong assumption of independence class features that it makes. It is near to impossible to find such data sets in real life.
##### Applications of Naive Bayes Algorithm :
1- Naive Bayes is widely used for text classification
2- Another example of Text Classification where Naive Bayes is mostly used is Spam Filtering in Emails
3- Other Examples include Sentiment Analysis ,Recommender Systems etc


#### SOURCES 
```
https://medium.com/@srishtisawla/introduction-to-naive-bayes-for-classification-baefefb43a2d
https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
https://machinelearningmastery.com/naive-bayes-for-machine-learning/
```

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