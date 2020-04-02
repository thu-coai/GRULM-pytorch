from scipy.stats import truncnorm
from operator import mul
import numpy as np
from functools import reduce

def truncated_normal(*size, threshold=2):
	all_size = reduce(mul, size, 1)
	values = truncnorm.rvs(-threshold, threshold, size=all_size)
	return np.reshape(values, size)
