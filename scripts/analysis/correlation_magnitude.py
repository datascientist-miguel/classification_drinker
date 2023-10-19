import numpy as np
def correlations_magnitude(data):
    corr = data.corr()
    correlation_sum = np.sum(corr.abs(), axis=1) - 1
    correlation_sum = np.sqrt(correlation_sum)
    correlation_sum = correlation_sum.sort_values(ascending=False)
    
    return correlation_sum