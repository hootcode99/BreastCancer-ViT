# Lung Disease Classifier  (work-in-progress)

I am currently developing a Convolutional Neural Network to take in chest X-rays and output probabilities for 14 different types of lung disease.

I am leveraging the NIH Chest X-ray Dataset which can be found here: 
https://www.kaggle.com/datasets/nih-chest-xrays/data/data 

With this project, I'm looking to learn more about PyTorch Lightning, Data Pre-Processing, and Optuna Hyperparameter Optimization.

I have already done some pre-processing to downsize the images to a size more appropriate local training and prototyped a model on that input size. Currently, I'm developing a notebook that will transform & organize the data labels to be more usable with PyTorch. The data labels did not come in a useful format and some data needs to be excluded due to quality issues or class over-representation. I'm leveraging Pandas to rectify the labeling format and will be using Pillow to exclude some of the data.