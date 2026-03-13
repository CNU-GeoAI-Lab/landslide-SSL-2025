import numpy as np
from scipy.interpolate import griddata
from copy import deepcopy


def fill_nan_nearest_2d(data):
    """
    Fill NaN values with the nearest non-NaN value in each 2D slice.
    data: array of shape (N, H, W). For each of the N slices, replace NaN with nearest valid value.
    Returns: array of same shape (modified in-place on a copy).
    """
    out = data.copy()
    for i in range(data.shape[0]):
        sl = out[i]
        valid = ~np.isnan(sl)
        if not np.any(valid) or np.all(valid):
            continue
        points_valid = np.argwhere(valid)
        values_valid = sl[valid]
        points_nan = np.argwhere(~valid)
        filled = griddata(points_valid, values_valid, points_nan, method="nearest")
        # Fallback: griddata can sometimes return NaN at edges; replace any remaining with global mean
        if np.any(np.isnan(filled)):
            filled = np.nan_to_num(filled, nan=np.nanmean(values_valid))
        out[i][~valid] = filled
    return out

class MinMaxScaler():
    def fit(self, x):
        self.min = x[~np.isnan(x)].min()
        self.max = x[~np.isnan(x)].max()        
        
    def transform(self, x):
        x_scaled = (x - self.min)/(self.max - self.min)
        if np.isnan(np.sum(x_scaled)):            
            x_scaled = np.nan_to_num(x_scaled, nan=0) #T,ODO: try different value for replacing Nan value 
        return x_scaled
    
    def reverse(self, x):
        return x*(self.max - self.min)+self.min   #T,ODO: didn't consider nan or zero for outside of the field

class StandardScaler():
    def fit(self, x):
        x_clean = x[~np.isnan(x)]
        self.mean = x_clean.mean()
        self.std = x_clean.std()
        # Avoid division by zero
        if self.std == 0:
            self.std = 1.0
        
    def transform(self, x):
        x_scaled = (x - self.mean) / self.std
        if np.isnan(np.sum(x_scaled)):            
            x_scaled = np.nan_to_num(x_scaled, nan=0) #T,ODO: try different value for replacing Nan value 
        return x_scaled
    
    def reverse(self, x):
        return x * self.std + self.mean   #T,ODO: didn't consider nan or zero for outside of the field

class RobustScaler():
    def fit(self, x):
        x_clean = x[~np.isnan(x)]
        self.median = np.median(x_clean)
        q75, q25 = np.percentile(x_clean, [75, 25])
        self.iqr = q75 - q25
        # Avoid division by zero
        if self.iqr == 0:
            self.iqr = 1.0
        
    def transform(self, x):
        x_scaled = (x - self.median) / self.iqr
        if np.isnan(np.sum(x_scaled)):            
            x_scaled = np.nan_to_num(x_scaled, nan=0) #T,ODO: try different value for replacing Nan value 
        return x_scaled
    
    def reverse(self, x):
        return x * self.iqr + self.median   #T,ODO: didn't consider nan or zero for outside of the field

class Multi_data_scaler():
    def __init__(self, multi_features):
        self.multi_features = multi_features
        self.num_channel = multi_features.shape[3]
        self.scalers = [MinMaxScaler() for _ in range(self.num_channel)]

    def multi_scale(self, test_data):
        test_data_scaled = deepcopy(test_data)
        for i in range(self.num_channel):
            self.scalers[i].fit(self.multi_features[:,:,:,i])
            test_data_scaled[:,:,:,i] = self.scalers[i].transform(test_data[:,:,:,i])
        
        return test_data_scaled

class Multi_data_standard_scaler():
    def __init__(self, multi_features):
        self.multi_features = multi_features
        self.num_channel = multi_features.shape[3]
        self.scalers = [StandardScaler() for _ in range(self.num_channel)]

    def multi_scale(self, test_data):
        test_data_scaled = deepcopy(test_data)
        for i in range(self.num_channel):
            self.scalers[i].fit(self.multi_features[:,:,:,i])
            test_data_scaled[:,:,:,i] = self.scalers[i].transform(test_data[:,:,:,i])
        
        return test_data_scaled

class Multi_data_robust_scaler():
    def __init__(self, multi_features):
        self.multi_features = multi_features
        self.num_channel = multi_features.shape[3]
        self.scalers = [RobustScaler() for _ in range(self.num_channel)]

    def multi_scale(self, test_data):
        test_data_scaled = deepcopy(test_data)
        for i in range(self.num_channel):
            self.scalers[i].fit(self.multi_features[:,:,:,i])
            test_data_scaled[:,:,:,i] = self.scalers[i].transform(test_data[:,:,:,i])
        
        return test_data_scaled