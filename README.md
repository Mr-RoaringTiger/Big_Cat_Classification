# Big Cat Classification

This project involves building and evaluating a deep learning model using Convolutional Neural Networks (CNN) for multiclass classification of lions, tigers, cheetahs, and leopards.

## Project Structure

The project includes two Jupyter Notebook files:
1. `big_cat_recognition.ipynb`: This notebook is used to train the CNN model and save the evaluation metrics (loss and accuracy).
2. `graph_and_prediction_big_cat.ipynb`: This notebook is used to load the saved model and evaluation metrics, plot graphs for loss and accuracy, and make predictions.

## Libraries Used

The following libraries are used in this project:

```python
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np
