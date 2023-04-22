#CP322 
# 9.1 - 9.5
- *C* is a nonnegative tuning parameter
	- Think of it as the budget for the amount the margin can be violated for *n* observations
- If *C* = 0, then there is no budget for violations to the margin
	- Margin is the dotted line around the hyperplane
- As *C* increases, the we become more tolerant of violations to the margin
	- As *C* decreases, we become less tolerant to violations of the margin
	- The margin **NARROWS**
- *C* is chosen during cross validation and controls the bias-variance trade off


- Objects that lie strictly on either side of a hyperplane do not determine its behaviour
	- Only objects that lie on or inside the margin affect the hyperplane
		- These are called **Support vectors**

- If there are many support vectors (meaning large C), then the model will have low variance but high bias
- If there are few support vectors (small C), the model will have high variance and low bias



- Support Vector Machine is a Support Vector Classifier than enlarges feature space using **kernels**
	- Kernerls are a function that returns the similarity of two vectors that are inputted
	- Kernels can be linear or polynomial (of degree *d*)
- Polynomial kernels lead to much more flexible decision boundaries
- When a support vector classifier is combined with a non-linear kernel, we get a **support vector machine**


# 10

### Neural Networks

- CNN - Convolutional Neural Networks (typically used for image classification)
- RNN - Recurrent Neural Networks (typically used for time series and sequences)

- A neural network takes an input vector of *p* variables and builds a non-linear function to predict a response

- p=4 predictors (X_1, X_2, X_3, X_4)
	- This is the *input layer*

- p is the count of input layer nodes, K is the count of hidden layer nodes

- Each input node is connected to the K number of nodes in the Hidden Layer
	- The hidden layers then all connect to the Output layer f(X)->Y

![[Pasted image 20230410185935.png | 400x350]]

- In early neural networks, *sigmoid* activation functions were favored 
	- ![[Pasted image 20230410190225.png | 300x100]]
	- Sigmoid function is the same function used in logistic regression to convert linear probabilities to values between 0 and 1
- Now, modern neural networks use the ReLU function 
	- Rectified Linear Unit

- Fitting a neural network requires estimating the unknown parameters
	- Typically for quantitative responses, squared-error los is used

- The more input layers and hidden layers, the closer the model gets to a "good solution" 
- In the case of the `MNIST` neural network that converts handwritten numbers to numeric values, it has two hidden layers (256 and 128 units; 256 in layer 1 for the greyscale values of each image pixel from 0 to 255 and 128 in layer 2 for something else) and 10 output layers for digits 0-9


- Ridge regularization and Dropout regularization are two regularization techniques used in deep learning
	- Ridge is a regularization that adds a penalty term to the loss function that prevents overfitting
		- Shrinks the coefficients of the model towards zero
	- Dropout randomly drops some of the neurons during training to prevent overfitting
		- Prevents interdependence between neurons

- Softmax activation function is often used in deep learning to normalize the output of a neural network
	- Scales numbers/logits into probabilities

- Cross entropy is used in deep learning to measure the difference between the predicted output and actual output

- feature map is a 3d array for images
	- contains 2 spatial axis and the 3rd dimension being the channel axis representing 3 colors (RGB)

### Convolutional Neural Networks

- CNNs that recognize images first identify low-level features such as small edges or patches of color
	- Then the low-level features are combined in the next layer to form high level features such as eyes, ears, etc
- The presence or absence of these features are what the CNN uses to output is decision

- CNNs use convolution layers and pooling layers to examine low and high level features
	- Convolution layers search for instances of small patterns in the image
	- Pooling layers downsample and select the small patterns to select the prominent subset


- A convolution layer is made up of many convolution filters
	- Each filter is a template of whether or not a particular feature is present in an image
	- Uses a convolution which basically repeatedly multiplies matrix elements and adds the results
	- ![[Pasted image 20230411151123.png | 320 x 200]]
	- If the convolved matrix someone resembles the convolution filter then it will have a large value, if not it will have a small value
- The convolved image *highlights regions of the original image that resemble the convolution filter*

- The submatrix of the convolution filter examines a section of the image in pixels
	- In order to increase the size of the submatrix, the padding must be increased


 - Pooling layers provide a way of condensing a large image into a smaller summary image
	 - Basically converts a 4x4 pixel image to a 2x2 pixel image
 - It also provides *location inheritance* where as long as one of the pixels in the block has a large value, the large value stays in the pooled, condensed matrix


- Data augmentation is a step in neural network training where a training image replicated and distorted in many different ways (stretched, shrunk, rotated etc) to 

- After pooling and convolving so the original image is just a few pixels, the 3d feature maps are *flattened*
	- This means the pixels in the feature map are treated as seperate units and fed into fully connected layers to be classified 


### Document Classification

- Featurizing a document is converting the raw text into a set of features that can be used as an input in a model
- The most common featurization is using a *bag-of-words*
	- If the dictionary contains M words, we create a vector of length M and score a 1 for every word present and 0 for every word not present

- If a bag of words is used that is length 10,000 words but only 100 of the 10,000 words are used that creates a 10,000 length vector with 100 1s and 9,900 0s
- This is a sparse matrix where most of the data is not used
	- In order to store a sparse matrix efficiently, we use *sparse matrix format* where instead of storing the whole vector (including all the 0s), we only store the indexes of the 1s


- The bag-of-words method summarizes a document based on the words used but not based on context, to used context there are two similar models:
	- *bag-of-n-grams* model, a bag of 2-grams model records the consecutive co-occurence of every distinct pair of words (ie "blissfully long" vs "blissfully short")
	- Treat the document as a sequence and use the context of the documents before and after


### Recurrent Neural Networks 
![[Pasted image 20230411225224.png | 400]]
- Classify things such as:
	- Book and movie reviews, newspaper articles, tweets (sequence and positions of words in a document capture narrative, theme, tone etc and can be used for topic classification, sentiment analysis, and language translation)
	- Time series of temperature, rainfaill, wind speed etc
	- Financial time series to track market indices, trading volumes, stock and bond prices
	- Recorded speech, musical recordings and other sounds for transcription or language translation
	- Handwriting such as doctors notes, zipcodes etc


- In RNNs the input is a *sequence*
	- Documents are a sequence $X = \{X_1, X_2, ..., X_L\}$ where each $X_l$ is a word

- Weight sharing is a practice used to reduce overfitting and decrease the number of parameters
	- Neurons in a neural network have the same filter applied to detect local features


- *Embedding space* is a lower dimensional space where high dimensional embeddings are projected???
	- Basically just a better way of storing words used in a vector because it takes up less space
	- Low dimension and high dimension refer to the number of features  (or dimensions) in the dataset
![[Pasted image 20230411232226.png|300]]
- *Embedding layers* are the hidden layers in an RNN that map information from a high dimensional space into a lower dimensional space
- Embedded layers are layers that are initialized with pre trained weights, fixing the weights during training so that the optimizer does not change them reduces overfitting and improves performance
	- This is called *weight freezing*

- *Long-term* RNNS refer to the ability of RNNs to capture long-term dependencies in sequential data
- Traditional RNNs have *short-term* memory
- LSTM RNN


- When looking at a time series prediction (for example predicted stock trading volumne) *auto correlation* is the correlation between two observation between two different points in a time series
	- Indicates that past values affect the current values
- *Lag* refers to the time gap between two values being considered for auto correlation

- A *straw man* refers to a simple prediction that can be used as a baseline for comparison


- Simple RNNs can share a lot in common with traditional *autoregression* (AR) linear models
- Below is the response vector and matrix of predictors for an autoregression model ![[Pasted image 20230412145341.png | 350]]
- **y** and **M** have T-L rows (where T is the total number of observations and L is the number of lagged values used as predictors)
- Fitting a regression of **y** on **M** results in 
	 ![[Pasted image 20230412145511.png|320]]

- The difference for a basic RNN and AR model is:
	- The RNN processes the sequence of words or data from left to right with the same weights (weight sharing)
	- The AR model treats all *L* elements of the sequence equally as a vector
		- This is a process called *flattening*

- *flattening* is the process of converting a multidimensional input into a one dimensional input
	- in CNNs a flatten layer is used between the convolutional layers and the dense layer (fully connected layer) to reduce feature maps to one vector

- The basic RNN covered in the notes covers the text sequence from left to right, there are RNNs that are *bidirectional* meaning that they cover the sequences in both directions

- *Seq2Seq* learning is when hidden units are thought to capture semantic meaning of sentences
	- Seq2Seq consists of a coder and decorder
	- Encoder takes a sequence of output data such as a sentence
	- Decoder then takes the vector representation and generates a sequence of output data such as a translated sentence



### When to Use Deep Learning
- Technically neural networks could be used for everything but take time and compuation power
	- For some examples, linear models outperforms neural networks with easy implementation and are way easier to understand
- *Occam's razor principle:*  if two models perform equally well then the simpler model should be preferred over the more complex one


### Fitting a Neural Network
- *Local minimum* is a point where a function takes a minimum value in a small neighborhood
- *Global minimum* is a point where a function takes a minimum value over the entire domain
![[Pasted image 20230412173710.png | 370]]

- There are two strategies to prevent overfitting:
	- *Slow learning*: model fit is somewhat slow iterative fashion using *gradient descent*. The fitting process is stopped when overfitting is detected
		- *Gradient descent* is an optimization algorithm used to minimize a function by iteratively adjusting the values of the parameters in the direction of the negative gradient of the function 
			- negative gradient is a vector that points in the direction
	- *Regularization*: penalties are imposed on the parameters, usually lasso or ridge 

#### Gradient
- A gradient is a vector of partial derivatives of a function 
- A gradient points in the direction of steepest ascent of the function
	- Basically, a gradient points in the direction of the greatest increase of the function and its magnitude is the slope in that direction

### Backpropagation
- An algorithm used to train neural networks
- Computes the gradient of the loss function of the loss function 
- Propagates the error backward through the network, starting at the output layer and adjusts each of the weights in the network as it goes back
	- Uses the chain rule of calculus to do so


### Regularization and Stochastic Gradient Descent
- For large amounts of data, gradient descent can be slow
- To speed it up, we can take a small fraction or *minibatch* of the observations and compute a gradient step
	- This process is known as *stochastic gradient descent*

- Epochs are the number of times a full training set has been processed
	- Early stopping can be used as a form of regularization


### Dropout Learning
- new form of regularization
- Randomly remove a fraction of the units in a layer when fitting the model ($\phi$)
- The surviving units stand in for those missing and the weights of the remaining units are scalled up by a factor of 1 / ( 1 - $\phi$) to compensate
- This prevents the nodes from being over-specialized


### Network Tuning
- There are a number of choices that effect network performance:
	- *The number of hidden layers and units per layer*
		- Modern thinking is that the number of units per hidden layer can be large and overfitting can be controlled with regularization
	- *Regularization tuning parameters*
		- Dropout rate $\phi$, and the strength $\lambda$ of lasso and ridge regularization are typically set seperately at each layer
	- *Details of stochastic gradient descent*
		- Batch size, number of epochs, and if used, details of data augmentation

### Interpolation and Double Descent

TLDR:
- Interpolation is the ability of a model to fit the training data perfectly
- Double descent is a phenomenon in which the test error of a model first decreases, then increases, then decreases again as the number of model parameters increases

- It is generally not a good idea to *interpolate* the training data - meaning to get 0 training error


# Natural Language Processing (CH11)
- focuses on developing techniques to produce machine-driven analyses of text

### Applications of NLP
- Automatic summarization
- Machine translation
- Named entity recognition
- Relationship extraction
- Sentiment analysis
- Speech recognition
- Topic segmentation

### Common Terminology
- Corpus: a collection of written texts that serve as our datasets
- Token: a string of contiguous characters between two spaces or between a space and punctuation marks
	- Can be text, integer, real, or number with a colon 
- Tokenization: The process of converting a text into tokens
- Text object: a sentence, phrase, word, or article


### Text Preprocessing
- Comprised of three steps
	- Noise removal
	- Lexicon normalization
	- Object standardization


#### Noise Removal
- Any piece of text that is not relevent to the data and end output is *noise*
	- example: is, am, the, of, URLs, social media entities (hashtags or mentions), punctuations, and industry specific words
- This step deals with removal of all types of noise entities in text

- A general approach for noise removal is to prepare a dictionary of noisy entities and iterate the text object, removing the characters in the dictionary
- Another apporach is to use regular expressions while dealing with special patterns of noise


#### Lexicon Normalization
- Another type of text noise is multiple representations of a single word
	- example: "play", "player", "played", "plays", "playing" ; Although they all are different contextually, they mean the same thing
- Lexicon normalization converts all disparities of a word into their normalized form (also known as the *lemma*)
- Normalization is a pivotal step for feature engineering with text as it converts the high dimensional features (having N many different features) to the low dimensional (1 feature) space


##### Example of normalization
- We need to "normalize" terms
- Indexed text and query terms must have the same form
- We want to match **U.S.A** and **USA**
- We implicitly define equivalence classes of terms
	- example: deleting periods in a term
- Alternative: asymmetric expansion:
	- Enter: **window** Search: **window, windows**
	- Enter: **windows** Search: **Windows, windows, window**
	- Enter: **Windows** Search: **Windows**
- Potentially more powerful, but less efficient


- The most common lexicon normalization practices are:
	- *Stemming*
	- *Lemmatization*

### Lemmatization
- Organized and step by step process of obtaining the root form of a word
	- Uses vocabulary (dictionary and importance of words) and morphological analysis (word structure and grammar relations).
- Reduce inflections or variant forms to the base form
	- *am, are, is* -> *be*
	- *car, cars, car's, cars'* -> *car*
- "the boy's cars are different colors" -> "the boy car be different color"
- Lemmatization: have to find the correct dictionary headword form

#### Morphology
- Morphemes: The small meaningful units that makeup words
- **Stems**: The core meaning-bearing units
- **Affixes**: Bits and pieces that adhere to stems
	- Often with grammatical functions


### Stemming
- Stemming is a rudimentary rule-based process of stripping the suffixes ("ing", "ly", "es", "s" etc) from a word
- Reducing terms to their stems in information retrieval
- *Stemming* is crude chopping of affixes
	- Language dependent
	- example: **automate(s), automatic, automation** all reduced to **automat**

- "For example compressed and compression are both accepted as equivalent to compress" -> "For exampl compress and compress ar both accept as equival to compress"


### Other text preprocessing steps
- Encoding-decoding noise
- Grammar checker
- Spelling correction etc


### Text to Features (Feature Engineering on text data)
- To analyze preprocessed data, it needs to be converted into features
- Depending upon the usage, text features can be constructed using some techniques:
	- Syntactical parsing
	- Entities / N-grams / word-based features
	- Statistical features
	- Word embeddings


### Syntactic Parsing
- Syntactical parsing involves the analysis of words in the sentence of grammar and their arrangement in a manner that shows relationships between words
- Dependency Grammar and Part of Speech tags are the important attributes of text syntactic

- Dependency Trees
	- Sentences are composed of some words sewed together. The relationship among the words in a sentence is determined by the basic dependency grammar

- Dependency Grammar
	- It is a class of syntactic text analysis that deals with (labeled) asynmmetrical binary relations between two lexical items (words). Every relation can be represented in the form of a triplet (relation, governor, dependent)![[Pasted image 20230413153158.png | 400]]

- The tree shows that "submitted" is the root word of this sentence, and is linked by two sub-trees (subject and object subtrees)
- Each subtree is a dependency tree with relations such as ("Bills" <-> "ports" <\by\> "proposition" relation), ("ports" <-> "immigration" <\by\> "conjugation relation")


#### Speech Tagging
- Apart from the grammar relations, every word in a sentence is also associated with a part of speech (pos) tag (nouns, verbs, adjectives, adverbs etc)
- The pos tags define the usage and function of a word in the sentence
![[Pasted image 20230413155129.png | 375]]
![[Pasted image 20230413155217.png | 375]]

#### POS Tagging 
- Words often have more than one POS:
	- The *back* door = JJ
	- On my *back* = NN
	- Win the voters *back* = RB
	- Promised to *back* the bill = VB
- The POS tagging problem is to determine the POS tag for a particular instance of a word

![[Pasted image 20230413160528.png | 320]]
- Uses:
	- Text-to-speech (how do we pronounce words like "lead")
	- Can write regexps like (Det) Adj* N+ over the output for phrases, etc
	- As input to or to speed up a full parser
	- If you know the tag, you can back off to it in other tasks


- Important uses of Part of Speech tagging:
	- Word sense disambiguation:
		- Some language words have multiple meanings depending on context
		- "Please book my flight for Delhi"
		- "I am going to read this book on the flight"
		- "Book" is used in a different context, however, the part of speech tag for both cases is different. In the first sentence, "book" is used as a verb and in the second it is a noin
		- Lesk Algorithm can be used for similar purposes
	- Improving word based features
		- A learning model could learn different contexts of a word when using words as the features, however, if the part fo speech tag is linked with them then the context is preserved
		![[Pasted image 20230413160938.png | 330]]
	- Normalization and Lemmatization
		- POS tags are the basis of lemmatization process for converting a word to its base form (lemma)
	- Efficient stopword removal
		- POS tags are also useful in the efficient removal of stopwords
		![[Pasted image 20230413161509.png | 330]]

### Entity Extraction (Entities as Features)
- Entities are defined as the most important chunks of a sentence - noun phrases, verb phrases or both
- Entity Detection algorithms are generally ensemble models of rule based parsing, dictionary lookups, pos tagging and dependency parsing
- The applicability of entity detection can be seen in the automated chat bots, content analyzers, and consumer insights

- Topic Modelling & Named Entity Recognition are the two key entity detection methods in NLP

#### Named Entity Recognition (NER)
- The process of detecting the named entities such as person names, location names, company names etc from the text is called as NER
	- For example: 
	- Sentence: "Sergey Brin, the manager of Google Inc. is walking in the streets of New York"
	- Named Entities: ("person": "Sergey Brin"), ("org": "Google Inc."), ("location": "New York")

- NER Model consists of three blocks
	- *Noun phrase identification*: This step deals with extracing all the noun phrases from a text usng dependency parsing and part of speech tagging
	- *Phrase classification*: This is the classification step in which all the extracted noun phrases are classified into respective categories (locations, names, etc)
		- Google maps API can be used for locations and DBpedia and Wikipedia can be used to identify persion names or company names
		- You can also make lookup table and dictionaries by combining information from sources
	- *Entity disambiguation*: Sometimes it is possible that entities are misclassified, hence creating a validation layer on top of the results is useful
		- Knowledge graphs can be used in the validation layer (Google Knowledge Graph, IBM Watson, Wikipedia)


#### Topic Modeling
- Topic modeling is a process of automatically identifying the topics present in a corpus, it derives the hidden patterns among the words in a corpus in an unsupervised manner
	- Topics are defined as "repeating patterns of co-occuring terms in a corpus"
	- A good model can identify "health", "doctor", "patient", "hospital" for a topic "Healthcare" AND "farm", "crops", "wheat" for the "Farming" topic


### N-Grams as Features
- A combination of N-words together is called N-grams
- N-grams (N>1) are generally more informative than words (unigrams) as features
- Bigrams (N=2) are considered the most important features of all the others


- Vectorization is jargon for a classic approach of converting input data from its raw format (text) into vectors of real numbers which is the format that ML models support


### Bag of Words
- Bag of Words involves 3 operations:
	- *Tokenization*: first, the input text is tokenized; a sentence is represented as a list of its constituent words
	- *Vocabulary creation*: Of all the obtained tokenized words, only unique words are selected to create the vocabulary and then sorted in alphabetical order
	- *Vector creation*: Finally, a sparse matrix is created for the input, out of the frequency of vocabulary words. In the matrix, each row is a sentence vector whose length (the columns of the matrix) is equal to the size of the vocabulary


### TD - IDF
- TF - IDF is a weighted model commonly used for information retrieval problems
- It aims to convert the text documents into vector models on the basis of the occurence of words in the documents without considering the exact ordering

- *Term Frequency* (TF) for a term "t" is defined as the count of a term "t" in a document "D"
- TF = Frequency of word in a document / Total number of words 

- *Inverse Document Frequency* (IDF) For a term is defined as the logarithm of the ratio of total documents available in the corpus and the number of documents containing the term T
- IDF = log( Total number of documents / Documents containing word W)

- TF - IDF formula gives the relative importance of a term in a corpus given the formula below
- $W_{i,j} = tf_{i, j} * log(N/df_i)$
- where tf is TF, df is IDF, and N is total number of documents


### Text Matching / Similarity
- One of the important areas of NLP is the matching of text objects to find similarities
- applications can include: automatic spelling correction, data de-duplication, genome analysis


#### Levenshtein Distance
- Distance between two strings is defined as the minimum number of edits needs to transform one string to another

- Edit operators allowed are insertion, deletion, or substitution of a single character

#### Phonetic Matching
- Takes a keyword as an input (persons name, location name etc) and produces a character string that identifies a set of words that are roughly phonetically similar
	- Similar pronounciations, sound patterns etc
- This helps search large text corpora, correct spelling errors and match relevant names

#### Cosine Similarity
- When the text is represented as a vector, a general cosine similarity can also be applied to measure vectorized similarity
- The cosine similarity of two documents ranges from 0 to 1
	- If the cosine similarity score is 1, it means two vecots have the same orientation
	- The closer the value gets to 0 it indicates the documents have less similarity


- *Text summarization* - given a text article or paragraph, summarize it automatically to produce the most basic and relevant sentences in order
- *Machine translation* - Automatically translate text from one human language to another using grammar, semantics, information about the real world etc
- *Nature Language Generation and Understanding* - Converting information from computer databases or semantic intents into readable human language is called **language generation**. Converting chunks of text into more logical structures for computer programs to manipulate is called **language understanding**
- *Optical Character Recognition*: Given an image representing printed text, determine the corresponding text
- *Document to Information*: This involves parsing textual data in documents (websites, files, pdfs, and images) in an analyzable and clean format


### Naive Bayesian Classifier
- Determine the most probable class label for each object
- Based on the observed object attributes
- Naively assumed to be conditionally independent of each other
	- Example: 
	- based on the objects attributes (shape, color, weight)
	- given that an object is {spherical, yellow, < 60 grams} it may be labelled as a tennis ball
- Class label probabilities are determined using Bayes' Law
- Input variables are discrete
- Output: 
	- Probability score - proportional to true probability
	- Class label - based on the probability score
- Preferred method for many text classification problems
- Use cases:
	- Spam filtering
	- Fraud detection

- Categorical variables are generally better suited for Bayesian Classifier than a logistic regression
![[Pasted image 20230413202329.png | 400]]



- Some considerations for the Naive Bayesian Classifier
- *Numerical Underflow*
	- Resulting from multiplying several probabilities near zero
	- Preventable by computing logarithm of the products
- Zero probabilities due to unobserved attribute/classifier pairs
	- Resulting from rare events
	- Handled by smoothing (adjusting each probability by a small amount)



- Problems that can arise with vector space models:
	- *Synonymy*: many ways to refer to the same object, example: car and automobile
		- Leads to poor recall
	- *Polysemy*: most words have more than one distinct meaning, example: model, python, chip
		- Leads to poor precision
- *Latent Semantic Indexing* was proposed to address these problems


#### Latent Semantic Analysis
- Consists of four basic steps
	- Convert matrix entires to weights
	- Rank-reduced *Singular Value Decomposition* (SVD) performed on matrix
		- All but the k highest singular values are set to 0
		- Produces k-dimensional approximation of the original matrix (in least-squares sense)
		- This is the "*semantic space*"
	- Compute Similarities between entities in semantic space (usually with cosine)

- SVD
	- Unique mathematical decomposition of a mtrix into the product of three matrices
		- Two with orthonormal columns
		- One with singular values on the diagonal
	- Tool for dimension reduction
	- Similarity measure based on co-occurence
	- Finds optimal projection into low-dimensional space

- SVD can be viewed as a method for rotating the axes in n-dimensional space, so that the first axis runs along the direction of the largest variation among the documents
	- The second dimension runs along the direction with the second largest variation
- Generalized least squares method


# CH12

- In *unsupervised learning*, we are not interested in prediction, because we do not have an associated response variable Y
- Instead, the goal is to discover interesting things about the feature set $X_1, X_2,... X_p$

- *Unsupervised learning*: type of machine learning where an algorithm learns to recognize patterns or hidden structures in data without labels or guidance

- Unsupervised learning is often performed as part of an *exploratory data analysis*
	- *Exploratory data analysis* (EDA) is an approach to analyze data and gain insights into their key characteristics, structure, and relationships between variables


### Principle Components Analysis
- *Principle components* allow us to summarize a set with a smaller number of representative varaibles
- The principle component directions are *highly variable*
	- Meaning there is a large amount of variance in the dataset
	- The directions also define lines and subspaces that are as close as possible to the data cloud
- *data cloud* is a visualization of the data in the reduced dimensional space, defined by the principle components
- *Principle Components Analysis* (PCA) refers to the process by which principal components are computed and the subsequent use of these components in understanding the data
	- Unsupervised approach
	- Serves as a tool for data visualization and data imputation (filling in missing values in a matrix)


#### What are principal components
- Suppose we want to visualize *n* observations on *p* features
- If *p* is large, then there will be too much data to examine and each subset of data will be insignificant on the whole understanding
- PCA fixes this
- It finds a low-dimensional representation of a data set that contains as much as possible of the variation
- Seeks a small number of dimensions that are as interesting as possible
	- *interesting* is measured by the amount that the observations vary along each dimension

- The *first principal component* of a set of features ($X_1, X_2,... X_p$) is the normalized linear combination of the features
![[Pasted image 20230414140121.png | 275]]
- We refer to the elements $\phi_{11}, ..., \phi_{p1}$ as the *loadings* of the first principle component
	- Together, loadings make up the principle component loading vector $\phi_1 = (\phi_{11} \phi_{21} ... \phi_{p1})$$
- We need to contrain the loadings so that their sum of squares is equal to one, otherwise the values could result in extremely large variance

- The set of linear combinations of loadings ($z_{11}, z_{21}, ... z_{n1}$) are the *scores* of the first principle component 
	- **principal component scores**


- The loading vector $\phi_1$ with the elements (loadings) $\phi_{11},\phi_{21}, ..., \phi_{p1}$ define a direction in feature space
- If we project the *n* data points $x_1, ..., x_n$ onto this direction than the projected values are the **principal component scores** 


- After finding the first principal component, we can find the *second principal component*
- The *second principal component* is a linear combination of $X_1, ..., X_p$ features that has maximal variance out of all the linear combinations that are uncorrelated with $Z_1$ (where $Z_1$ is the first principal component)


- Once we have the principal components, we can plot them against each other to produce a low-dimensioanl view of the data


- For actual applications each loading vector places different weights on related parameters, for example:
	- The USArrests data set has {Assault, Murder, Rape andUrbanPop}
	- The first loading vector places similar weights on Assault, Murder, and Rape but less weight on UrbanPop
	- The second loading vector places more weight on UrbanPop and less on the rest of the parameters
![[Pasted image 20230414144447.png | 400]]
- This figure is a *biplot* because it displays the principal component scores and the principal component loadings



- Principal components provide low-dimensional linear surfaces that are closest to the observations
- The first principal component is a line in p-dimensional space that is closest to the *n* observations
- This provides a good summary of the data

- The first two principal components provide  span a plane that is closest to the *n* observations
![[Pasted image 20230414145534.png | 350]]
- The first 3 principal components span a three-dimensional place that is closest to the *n* observations and so on


- *Proportion of variance explained* (PVE) is the amount of information that is lost when projecting a data set onto the principal components
- The *total variance* present in a data set is the statistical measure that shows variability in the data

- PVE can be calculated with $1 - RSS/TSS$
- where TSS represents the total sum of squared elements of *X* and RSS represents the residual sum of squares of the *M*-dimensional approximation 

##### Scree Plot
![[Pasted image 20230414153013.png | 350]]


- The results obtained from PCA will depend on whether or not the variables have been scaled individually
	- This is in contrast to models such as linear regression, where scaling the variables has no effect
	- Multiplying a factor by *c* in a linear regression will simply lead to multiplication of the corresponding coefficient estimate by a factor of 1/*c* which has no substantive effect on the model


- Below is what happens when variables are scaled or not
![[Pasted image 20230414153236.png | 375]]

- Flipping the sign of a loading vector (+) to (-) or vice versa, has no effect on the loading vectors and the directions do not change
	- Same effect happens with the score vectors (scores of the nth principal component)



#### Deciding how many principal components to use
- In general, a *n* x *p* matrix **X** has min(*n* - 1, *p*) distinct principal components
- Typically, the number of principal components needed are decided by examing a *scree plot*
	- This is done by examing the plot and looking for the *elbow* of a scree plot, that is where the slope of the curve changes



### Missing matrix values and completion
- Principal components can be used to *impute* the missing values
	- This is a process known as *matrix completion*
- Matrix completion is part of the process of powering *recommender systems* (algorithms that recommend content, ads, etc to users) 
	- page 522 textbook



### Clustering Methods
- *Clustering* refers to techniques to find *subgroups* or *clusters* in a data set
- Both clustering and PCA simplify data with summaries but:
	- PCA looks to find low dimensional representations of the observations that explain a good fraction of the variance
	- Clustering looks to find homogenous subgroups among the observations

- Applications of clustering arise in marketing
- Example:
	- You have a large data set of individuals and their median household income, occupation, distance to nearest urban area etc
	- You want to perform *market segmentation* by identfying subgroups of people who might be more receptive to a particular form of advertising or more likely to purchase a product 

- There are two popular methods for clustering: *K-means clustering* and *hierarchical clustering*
	- In *K-means clustering* we do not know in advance how many clusters we want
	- we end up with a tree-like representation of observations called a *dendrogram* that allows us to view clusterings


### K-means Clustering
![[Pasted image 20230414160137.png | 400]]
- *Within cluster variation* is a measure that represents the amount of variation or dispersion of data points
	- The lower the variation, the more compact and homogenous the cluster, the higher the more spread out and homogenous

- The most common method to defne within cluster variation is the *Euclidean distance*

##### Algorithm
1. Randomly assign a number, from 1 to *K*, to each of the observations. These serve as the initial cluster assignments for the observations
2. Iterate until the cluster assignments stop changing
	1. For each of the *K* clusters, compute the cluster *centroid*. The kth cluster centroid is the vector of the *p* feature means for the observation of the kth cluster
	2. Assign each observation to the cluster whose centroid is the closest (where *closest* is defined using Euclidean distance)

- As long as this algorithm runs, the clustering obtained will continually improve until the result no longer changes (meaning a *local optimum* has been reached)
- Because the algorithm finds the local rather than global optimum, the results obtained will depend on the intial (random) cluster assignment of each observation in Step 1 of the algorithm

![[Pasted image 20230414161020.png | 450]]


### Hierarchical Clustering
- One potential disadvantage of K-means is that it requires to use pre-specify the number of clusters K
- *Hierarchical clustering* is an alternative approach that does not require a choice of K
- *Hierarchical clustering* produces a *dendrogram*
![[Pasted image 20230414161314.png]]
- The height of the fusion indicates how similar vertical observations are
![[Pasted image 20230414161703.png | 425]]

##### Algorithm
1. Begin with n observations and a measure (such as Euclidean distance) of all the (n 2) = n(n-1)/2 pairwise dissimilarities. Treat each observation as its own cluster
2. For i = n, n - 1, ..., 2:
	1. Examine all pairwise inter-cluster dissimilarities among the i clusters and identify the pair of clusters that are most similar. Fuse these two clusters. The dissimilarity between two clusters indicates the height in the dendrogram at which the fusion should be placed
	2. Compute the new pairwise inter-cluster dissimilarities among the i - 1 remaining clusters 


##### The most commonly-used types of linkage in hierarchical clustering
- **Complete**: Maximal intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B. Record the largest of these dissimilarities
- **Single**: Minimumal intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B, and record the smallest of these dissimilarities. Single linkage can result in extended, trailing clusters in which single observations are fused one-at-a-time
- **Average**: Mean intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B
- **Centroid**: Dissimilarity between the centroid for cluster A (a mean vector of length *p*) and the centroid for cluster B. Centroid linkage can result in undesirable *inversions*
![[Pasted image 20230414194709.png]]


- Single linkage is the most popular
- Average and complete are preferred over single because they yield more balanced dendrograms
- Centroid linkage is used in genomics but suffers the drawback of *inversion*
	- *inversion* is where two clusters are fused at a height below either of the individual clusters in the dendrogram leading to visualization difficulties


#### Choice of Dissimilarity Measure
- *Euclidean distance* is what is covered so far
- *correlation-based distance* considers two observations to be similar if their features are highly correlated
- The choice of dissimilary measure is very important, it has a strong effect on the dendrogram 


### Practical Issues in Clustering
#### Small Decisions with Big Consequences
- In order to perform clustering, some decisions must be made:
- Should the observations or features be standardized?

- In the case of hierarchical clustering:
	- What dissimilarity measure should be used?
	- What type of linkage should be used?
	- Where should we cut the dendrogram in order to obtain clusters?

- In the case of K-means clustering, how many clusters should we look for in the data

#### Validating the Clusters Obtained
- Any time clustering is performed, how do we know the clusters best represent the data and are not just noise? There is no clear decision on the best approach


- Both K-means and hierarchical clustering will assign each observation to a cluster, however this is not always the best option
- If there are outliers K-means and hierarchical may lead to distorted outputs, *mixture models* solves that problem