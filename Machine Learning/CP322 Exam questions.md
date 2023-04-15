## CH1

- Types of machine learning slide on slides ch 1 is important
	- Continuous target variables means use a regression etc

- Choose the right algorithm given a specific problem (also chapter 1 slides)

- Know the steps to fit each specific ML model
	- Cleaning and editing
	- Determining data types
	- Data size reduction
	- Examination of data distributions and regularization

- Know when standardization is required for an algorithm

- Why do we create training and test sets
	- What is the advantage


- What are the hyperparameters for a particular model you need to know
	- What is the impact of increasing or decreaseing the parameter
	- what is the impact of the exact value on a model


- **Bias variance trade off **
	- Know it for every model
	- Impact on bv trade off as model parameters change
	- Using test data?

- In a give scenario know which model is overfitted/underfitted

- Given python code know what it does or outputs

- Precision, recall, accuracy and F1 formulas
	- ROC CURVE TOO how to make judgements on it


## CH2

- No graph creation, look at the graph and make assumptions
	- Overfitted or underfitted
	- 2d and 3d

- Know decision boundaries of the models

- Know the basic formulas of models

- Know how to write a regression equation
	- Know how to calculate regression given a specific value

- KNN 

- **trade off**

- Difference between prediction accuracy and intersomething (interpretability?)

- Diagram on ch2 comparing different models flexibilty vs interpretability

- MSE vs flexibility graph with training vs test data slide 25

- Understanding what a decision boundary is 
	- Know over and under fitting regarding KNN decision boundaries

- What is a tuning hyperparameter for KNN
	- It is K (the count of nearest neighbors)
	- If it is just looking at 1 neighbor it is over fitted
	- Why is K odd? To break the "tie" 2 are in class A 2 in class B, which class should be chosen (this is a tie)


## CH3 

- Know forward and backwards selection
	- What is the RSS, how it will change
	- What will be the over all RSS for the model

- Dummy variable concept
	- Handling qualitative variables in a piece wise function


## CH4 

- Why we cant apply linear regression to a logistic problem

- Memorize the logistic regression formula (e^ Beta0 ..... / ....)
	- Slide 14

- The book ontent on Bayes theorem and discriminant analysis and the differentiation, how they can be applied 
	- Know discriminant analysis formula

- Pk formula for discriminant analysis - UNDERSTAND dont remember
- Dscrimiminant score formula slide 31 - UNDERSTAND


- When recall and precision are high what does it mean
	- Know which model is better based on these numbers

- Quadratic Discriminant Analysis decision boundaries 
	- Difference between linear and quadratic

- Naive Bayes is important
	- Understand the model and apply it to a situation


## CH5

- Cross validation vs LOOCV

- Kfold What is the size of k cross validation? 10

## Ch7

- Understand concept of reducing test error, bias and vias, why is regularization needed? to control bias and variance

- When to use lasso and ridge needs to be "very clear"



- When polynomial is given understand the parameters, hyperparameters for the polynomial models
	- How to control the hyperparameter

- What is the step function, why are we doing it

- Piecewise polynomials, switch models around a cut

- knot, degree, degrees of freedom of a cubic spline
	- Df is the number of predictors we use to predict the model


- How are nodes created and selected in a tree based method
- RSS for regression tree

- Gini index and cross entropy for a tree
	- Understand the decision tree exampe around slide 38

- Know the issues with the decision ree

- Bagging and boosting conceptual difference

- What is pruning and what is its purpose
	- Controls overfitting 
		- Controls bias and variance trade off