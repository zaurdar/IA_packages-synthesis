# ===========================
# Standard library
# ===========================
import os
import math
import random
from typing import Tuple, List, Optional

# ===========================
# Numerical
# ===========================
import numpy as np

# ===========================
# PyTorch
# ===========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===========================
# TensorFlow / Keras
# ===========================
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# ===========================
# ML utilities
# ===========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ===========================
# Visualization
# ===========================
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Reproducibility
# ===========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ===========================
# Device (PyTorch)
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
