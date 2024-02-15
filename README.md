# Lung Disease Classifier  (work-in-progress)

I am currently developing a CNN to consume chest X-rays and output diagnoses probabilities for 14 different types of lung disease. With this project, I'm looking to learn more about PyTorch Lightning, Data cleaning/preprocessing, and Optuna hyperparameter optimization.

I am leveraging the NIH Chest X-ray Dataset which can be found here: 
https://www.kaggle.com/datasets/nih-chest-xrays/data/data 

I have already completed preprocessing using Pillow to downsize the images to be compatible with local, single-GPU training and developed a prototype model on that input size. I've just finished developing a Jupyter Notebook that will leverage the Python Pandas module to transform the data labels from text to one-hot encodings to be more useful with PyTorch training. Currently, I'm exploring whether class imbalance is significant enough to be rectified and which methods I could leverage.
