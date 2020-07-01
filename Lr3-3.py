import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import numpy as np
import pandas as pd
import random
import warnings
from Lr3_2 import QRmethod
warnings.filterwarnings('ignore')

a, b = -2, 15
A = np.array([[a, 1], [b, 1]])
v = QRmethod(A, 0.0000000001)
print(v)