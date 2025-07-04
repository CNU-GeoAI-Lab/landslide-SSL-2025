import numpy as np

from copy import deepcopy

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