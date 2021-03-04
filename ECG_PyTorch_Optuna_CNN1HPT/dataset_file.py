import numpy as np 
import torch 


class dataset(): 
    def __init__(self, feature, label):
        self.features = feature 
        self.labels = label 

    def __len__(self):
        return len(self.features) 

    def __getitem__(self, idx):
        feature = self.features[idx]
        #  np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
        #feature = np.array(feature).reshape(self.features.shape[0], x_train.shape[1], 1)
        feature = np.transpose(feature, (1, 0)).astype(np.float) 
        label = self.labels[idx] 


        feature = torch.tensor(feature, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long) 

        return feature, label 
