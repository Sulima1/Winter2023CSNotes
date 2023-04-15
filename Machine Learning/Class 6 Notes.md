- Minimal Primary Key
	- unique identifier for each data point in a dataset
	- used to distinguish one data point from another and can be used to access, update or delete specific data points


1.  *Quantitative Continuous data* refers to data that can take on any value within a certain range, such as weight or height.
2.  *Quantitative Discrete data* refers to data that can only take on certain values, such as the number of siblings a person has.
3.  *Qualitative Ordinal data* refers to data that can be put into categories and the categories have a logical order or ranking, such as a person's education level (e.g. high school, associate degree, bachelor's degree, master's degree).
4.  *Qualitative Nominal data* refers to data that can be put into categories but the categories do not have a logical order or ranking, such as a person's eye color (e.g. brown, blue, green).


## K-fold Cross-validation in detail
- Divide data into K roughly equal-sized parts 
	- The first part of the list will be used for validation
	- The remaining 4 parts will be used for training
- Finds average MSE

- Example, if there are 10,000 models to train and k=10 folds
	- 1,000 models are used for validation
	- 9,000 are used for training