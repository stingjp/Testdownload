from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

# now we can load our data
"""
In this tutorial, we are going to use the Pima Indians onset of diabetes dataset. 
This is a standard machine learning dataset from the UCI Machine Learning repository. 
It describes patient medical record data for Pima Indians and whether they had an
onset of diabetes  five years.
"""

