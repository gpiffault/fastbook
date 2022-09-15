# 01 Intro

- Setting up the environment (software)
- Cats and dog classifier example (resnet34, bird resnet18 example in the course notebook 00)
- How does machine learning work overview (Arthur Samuel model)
- Deep learning jargon
- Other quick examples
  - Segmentation of dashcam images
  - Sentiment of a movie review (NLP)
  - Tabular data prediction
- Validation and test sets
  - When do we need a test set in addition to a validation set
  - How to choose for time series
  - How to choose when multiple images of the same person

# 02 Production

- What tasks deep learning is good at
- How to approach a problem (drivetrain approach)
- Example of gathering labeled data (photos of grizzly/black bears)
- Pitfalls of gathering data from the web
- DataLoaders example (type of data, get data items, get labels, split)
- Resize images methods (crop, squish, pad, random crop)
- Image data augmentation
- First train a quick model to help clean the data (plot_top_losses, ImageClassifierCleaner)
- Turning the model into a web app and deploy
- How to gradually replace a process with the model
- What to monitor when using a model in production
- Danger of feedback loops

# 04 MNIST Basics

- Basic operation on tensors
- Baseline MNIST model: distance (mean(abs(a-b))) to average of training sets
- NumPy arrays vs PyTorch tensors
- Broadcasting
- Stochastic gradient descent (SGD)
- Computing gradients with PyTorch
- Learning rate
- SDG end-to-end example with 3 parameters model (quad)
- Flatten 2D images to vectors (tensor.view())
- Matrix multiplication to make predictions
- Compute accuracy
- Accuracy is not a good loss function because it's derivative is always 0 (step function)
- Distance between predictions and targets as loss function
- Sigmoid (R -> [0,1])
- Mini batches, batch size (DataLoader)
- Whole SGD process by hand, then replace elements with fastai/PyTorch provided utilities
  - nn.Linear
  - DataLoaders
  - Learner (esp Learner.fit)
- Adding non linearity (nn.Relu)
- Combining Layers (nn.Sequential)
- Plotting accuracy
- Why more than one non linearity (deep models, better perf/total size)

# 08 Collaborative filtering

- Collaborative filtering (recommender system, latent factors)
- Cross table for representation
- Build a model architecture and use SGD to learn latent factors
- Data preprocessing, latent factors representation
- Embedding layer (array lookup compatible with gradient computation)
- Pytorch module to define and train a model
- Add sigmoid and bias
- Weight decay (L2 regularization) to fight overfitting
- More detailed Pytorch module definition (nn.Parameter)
- Interpretation of embedding and biases
- PCA for embeddings analysis
- fast.ai collab_learner
- Embedding distance
- Bootstrapping recomender systems
- Beware of representation bias and feedback loops
- Using embeddings in a neural net

# 09 Tabular

- Categorical embeddings strengths
- Pro and cons of deep learning vs random forest
- How to peek at tabular data
- Decision trees
- Tabular data preprocessing (categories order, dates, common transforms)
- Crafting a validation set (esp wrt test set)
- How to build a decision tree to explore the data
- Decision tree overfitting (big tree)
- How to deal with categories (do nothing, it works)
- Random forests
- How to spot overfitting (out of bag error)
- How to asses confidence of predictions (tree variance of preds)
- Feature importance
- Removing columns (low importance, redundancy)
- How a column impact prediction
- Data leakage (be mindful of data collection process)
- How to analyze/explain a single prediction
- Extrapolation problem with random forests
- Finding potential issues with validation/test set (try predicting the set of a row)
- Use a NN
- Boosting
