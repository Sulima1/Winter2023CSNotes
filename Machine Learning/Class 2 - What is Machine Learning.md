#CP322
- A model is a line of best fit that we are creating via ML
	- We need a correlation between certain predictors to make a prediction about some data set
	- ie: Sales = *f*(TV, Radio, Newspaper)
- Models do not need to be linear so long as the fit the plotted data points the best

- **Response variable** vs **predictors**
	- Response variable is Sales
	- Predictors are TV, Radio, Newspaper

- **Residuals** are the difference between the actual value (Y) and the prediction

 $$\epsilon = Y - f(x)$$
- Is the irreducible error - ie even if we knew f(x), we would still make erorrs in prediction, since at each X=x there is typically a distribution of possible Y values

- The actual value (Y) is the plotted value and the predicted value is the line of best fit generated
	- The difference between the two is called the **residual error**
	- The lower the residual error the better the model is


- The **linear** model is an important example of a parametric model
$$F_l (x) = \beta_0 + \beta_1 X_1 + ... \beta_n X_x$$

- **Multi-linear regression model** is when there is more than one predictor
	- ie $$income = f(education, seniority) + \epsilon$$
-  Models can also be a plane when there is more than 1 predictor

- Some plane models use a technique called **thin-plate spline** to fit a more flexible surface in three dimension

- When a model makes no errors it is known as **overfitting** (touches every point in the graph)
	- overfitting is a BAD characteristic as it is not general enough to use the model in many situations


- **Training error** vs **Test error**
	- Training error is the error rate of a model when applied to the data it was trained on
	- Test error, also known as generalization error, is the error rate of a model when applied to new, unseen data

- The goal of training a model is to minimize the test error, because a low test error indicates that the model is able to generalize well to new data.

- Interpretability of a model is how explainable the effect is 

