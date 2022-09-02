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
