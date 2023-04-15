
- *The primary difference for bagging and boosting in classification problems is*
	- **Bagging is a variance reduction technique while boosting is a bias reduction technique**

- Bagging (VARIANCE reduction)
	- Short for bootstrap aggregation
	- Create multiple samples (bags) of training data
	- Train a model on each bag
	- Reduces overfitting and variance BUT can be computationally expensive
- Boosting (BIAS reduction)
	- Iteratively train a model improving on the mistakes of the last each time
	- Combining multiple simple models to create a stronger one
	- Can be applied to many models, highly flexible BUT is sensitive to noisy and overfitting

- *Pruning a decision tree*
	- **Increases interpretability by reducing the number of splits**

- Decision tree
	- Supervised learning algorithm 
	- Recursively splits the data based on the feature that provides the most information given then stops when threshhold is met
	- Calculated using entropy, gini impurity or variance reduction
	- Easy to interpret but suffer from overfitting especially if tree is too complex
		- Pruning, ensemble methods and random forests fix this

- Entropy
	- measure of randomness or impurity of a node class distribution 
	- Less entropy = more information gain
- Gini Impurity
	- measure of randomness or impurity in a decision tree
	- Same as entropy determines information gain
- Variance reduction
	- Measure of the reduction in variance resulting from a tree split
	- Goal is to minimize variance

- Pruning
	- Technique used to prevent overfitting
	- Remove branches that do not improve data accuracy
- Ensemble Methods
	- Combining multiple methods to reduce variance 
	- Can be more than just trees
- Random forests
	- Combines mulitiples decision trees into a single forest
	- Handles complex data sets and reduces overfitting
	- averages voting predictions of all the trees

- Random forests are technically an ensemble method but Random forests can handle larger datasets at the risk of being more computationally expensive


- *Which of the following is not a common evaluation model for linear regression models*
	- **precision**

- Mean Squared Error (MSE)
	- Measure of the average squared distance between actual and predicted values
- Root Mean Squared Error (RMSE)
	- Square root of the MSE but easier to interpret
- R^2 
	- Measure of how well model fits the data 
	- Given as a proportion of variance 
- Mean Absolute Error
	- Measure of average absolute difference between predicted and actual values
	- Less sensitive to outliers than MSE and RMSE
- Residual Analysis
	- Analyzing the residuals to identify problem patterns or trends in the model
	- Helps identify nonlinearity or heteroscedasticity
- Cross-validation
	- Dividing the data into training and validation sets and evaluating the model on multiple subsets of data
	- Helps to identify overfitting

- *What type of response variable is used in logistic regression*
	- **binary**
	- Logistic regressions are often used to make binary classifications

- Logistic regression
	- Models relationship between predictor variables and binary response variables
	- Interpretable and able to handle non-linear relationships
	- Assumes linearity between predictor and log odds of the response maybe suffer from over or underfitting
	- can only handle classification with TWO OUTCOMES

- Nominal
	- categorical varible 
	- Gender, race eye color
- Binary
	- special case of nominal where there are only 2 possibilities
	- yes/no or true/false
- Continuous
	- Numerical value that can take on any value 
	- Age, height, temperature
- Ordinal (ORDer)
	- categorical variable with ranking or order
	- Education level (high school, college, grad)
	- Survey responses (stronly agree, disagree, neutral etc)

- *The number of position of knots in a regression spline are typically determined*
	- **determined by a method such as cross validation**

- Cross validation
	- Partitions the dataset into training and validation sets 
	- There are several types: [k fold cross validation, leave one out cross validation, statified cross validation]
	- Fights overfitting and can be used on different models BUT training k times can be computationally expensive

- K fold cross validation
	- data divided into k equally sized partitions and trained k times with each partitition used at the validation set once
- Leave one out cross validation
	- k is equal to the number of samples in the dataset

- Regression spline
	- Regression model that uses a piecewise function to fit a curve of data points
	- A knot is the value at which a regression spline switches from one formula to another


- *For a linear regression model, the coefficient of determination (R-squared) measures which of the following*
	- **goodness of fit of a model**

- R^2 is a measure of the goodness of fit of a model


- *Describe three real life situations where regression might be useful. Describe the response as well as the predictors. Is the goal of each application inference or prediction*
	- **Real estate agent wants to predict the sale price of houses. Response is the sale price of the house predictors would be number of bedrooms etc... The goal would be prediction**
	- **Healthcare worker wants to determine if there is a relationship between  weight, age and lifestyle factors with diabetes. Response is if they develop diabetes and predictors are age, weight, lifestyle etc... The goal is inference**


- Inference VS prediction
	- The goal of inference is the understand the relationship between variables 
	- What is the effect of X on Y

	- The goal of prediction is to make predictionts about new or future data points
	- What is the expected value of Y given X


- *Ridge regression reduces the variance of the model by adding a penalty term to the loss function*
	- **True**

- Ridge regression and Lasso add penalties to a model to prevent overfitting

- Ridge Regression
	- Adds a penalty term to the regression equation to shrink the coefficient towards zero
	- $Sum of Square Residuals + \Lambda \cdot sqrt(Slope)$
	- Useful for multicollinearity but requires tuning

- Lasso Regression
	- $Sum of Square Residuals + \Lambda \cdot abs(Slope)$
	- Useful for feature selection and model simplification
	- May overfit with many predictors

- Loss function
	- Function that calculates the difference between predicted and actual output values given a set of data


- *What is the purpose of feature scaling in polynomial regression*
	- **To speed up convergence of the optimization algorithm**
	- By scaling features we ensure each feature has similar impact

- Polynomial Regression
	- Regression analysis where the plotted line is to the nth degree of a polynomial

- Feature scaling
	- Technique to normalize a range of values
	- Standardization scales the data to have zero mean and unit variance
	- Normalization scales the values to be between 0 and 1

- Convergence
	- Occurs when the model has found the optimal parameters


- *Receiver operating characteristic (ROC) curve is a plot of the model's predicted probabilities against the observed outcomes*
	- **True**

- ROC curve
	- A way to evaluate the performance of binary models
	- Plots the TP rate against the FP rate
	- The area under the curve is often used to compare the performance of different models


- *Confusion matrix in classification problems is used to*
	- **summarize the performance of a classification model in terms of its accuracy and error rate**

- Confusion matrix
	- A table used to evaluate the performance of a model
	- TP, TN, FP, FN table
	- Can be used to calculate accuracy, precision, recall and F1 score

- Accuracy
	- A measure of correct predictions over total predictions
	- $(TP + TN)/(TP+FP+TN+FN)$
- Precision 
	- Measures the proportion of true positives and total number of positive predictionts
	- $TP/(TP+FP)$
- Recall
	- Proportion of true positives to the total number of actual positives
	- $TP/(TP+FN)$
- F1 Score
	- Harmonic mean of precision and recall
	- $((precision \cdot recall)/(precision + recall))$


- *When regularization is added to an underfitted model, which of the following statements is true in context of the error*
	- **It will decrease**
	- When regularization is added to an underfitted model, it will increase bias but decrease variance

 - Regularization
	 - Technique used in machine learning to prevent overfitting models by adding a penalty term
	 - Lasso or Ridge regression
	 - Too much regularization can lead to underfitting a model, effects should be validated through cross-validation

- *Which of the following is crucial for choosing between mutliple linear regression and KNN*
	- **The relationship between the predictors and response**
	- If the relationship is linear then MLR is appropriate, if not then KNN is more suitable
		- Linear regressions assume the relationship to be linear


- *Is this opinion correct?
	  "If we have few training observations and the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary"*
	- **Problems with statement: If there are few training observations, usng QDA may lead to overfitting the data and may not perform better than LDA depending on the data**


- Bayes Decision Boundary
	- A way to seperate two or more classes based on their features
	- Uses probabilities to determine which class is likely given features
	- Provides accurate classification but is hard to apply because we hardly know the true probability distributions of data
	
- When the distribution is not known, people often assume either QDA or LDA are used

- Quadratic Discriminant Analysis (QDA)
	- Models probability distribution of a class with a quadratic function
	- Works well when decision boundary is non linear
	- Assumes probability distributions to be quadratic

- Linear Discriminant Analysis (LDA)
	- Assumes features are normally distributed
	- Can be used for both binary and multiclass classification
	- Works well when assumptions are met but fails when they are not met (decision boundary is not linear)


- *Is this opinion correct?
	  "Lasso is less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance"*
	- **Yes it is correct. Lasso reduces variance but increases bias but if the use of lasso's reduction in variance outweighs the effects on bias it will result in higher prediction accuracy**

- *Which of the following statements is true in the context of ridge regression*
	- **Ridge regression can handle collinearity among predictor variables better than least squares regression**

- Collinearity
	- Two or more predictor variables in the regression model are highly correlated with each other
	- Ridge regression is designed to handle multicollinearity
		- When there is multicollinearity, the Ordinary Least Squares function estimates can become unstable and have high variance. Ridge regression adds the penalty to the OLS function and reduces how unstable it is

- Least Squares Regression
	- Method to find the best linear fit between predictors and response
	- Easy to implement and target BUT assumes errors are normal distributed
	  
- *Resampling methods in statistical analysis are used to reduce the size of the dataset*
	- **False**
	- Resampling methods do not affect the size of the data


- Resampling methods
	- sample of data is repeatedly drawn from a population and used to validate the performance of a model
	- cross-validation, bootstrapping


- *There is a small town where people are experience mysterious power outages. The towns council suspects that the outages are being caused by one of three potential culprits: Faulty power lines, overconsumption of electricy, or a malicious hacker. They have gathered labeled data where the outcomes are clearly defined on power consumption patterns, maintenance records for power lines and network logs for potential hacking attempts. Which supervised learning approach would you recommend to identify the cause of the power outages based on the available data sets*
	- **I would put decision tree because it handles classification and there are multiple outcomes. **


- *What are the advantages and disadvantages of the k-fold-cross-validation compared to validation set approach and LOOCV approach*
	- **k fold cross is more efficient with how it uses its data and provides a more stable model because it is tested multiple times BUT can be computationally expensive with a large K and not suitable for time series data**

- Validation set approach
	- Data split into training and validation
	- Training to train model and validation to evaluate performance
	- Size of validation set is small, leading to high variability in performance
	- Less efficient in terms of data when compares to cross validation

- LOOCV approach (leave one out cross validation)
	- divide data into n groups and use n-1 groups for training and remaining 1 for validation
	- maximizies training data to provide more accurate model
	- useful when the data set is small and when there is a need to avoid overfitting


- *Why is a weight assigned to each instance in the training data boosting*
	- **To give more importance to the instances that are difficult to classify**
	- This lets the model focus on where it is lacking in order to improve for the next iteration which is the process of boosting


- *In non-parametric bootstrap it is assumed that the data are normally distributed*
	- **In non parametric, no assumption is made about the data**

- Parametric Bootstrap
	- Assumes data follows a specific distribution 
- Non parametric assumes no specific distribtion


- *Is regulaization the process of reducing the number of features in data*
	- **False**


- *Bagging in the context of decision trees*
	- **Builds multiple trees and average their predictions to reduce variance**