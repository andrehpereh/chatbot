#import re
#import datetime
#import json
#import locale
#import keras
#import keras_nlp
#import torch
#import transformers
#from google.cloud import aiplatform
#from numba import cuda
#import os
#import tensorflow as tf
#import numpy as np
#import requests
#import time

# print(keras_nlp.__version__)
# print(tf.__version__)
# print(keras.__version__)

from util import get_model_paths_and_config
from config import Config

print(Config.MODEL_NAME)
model_info = get_model_paths_and_config(Config.MODEL_NAME)
print(model_info)