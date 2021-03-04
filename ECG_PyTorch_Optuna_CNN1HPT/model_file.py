
import numpy as np 
import torch 
import torch.nn as nn   
import torch.nn.functional as F 
from torch.autograd import Variable


class ecg_net(nn.Module):

    def __init__(self, trial):
        super().__init__() 

        self.features = nn.Sequential( 
                                    nn.Conv1d(
                                            in_channels=1, 
                                            out_channels=32, 
                                            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
                                            stride = trial.suggest_categorical('strides', [1, 2])
                                    ),   
                                    nn.ReLU(),
                                    nn.Conv1d(
                                            in_channels=32, 
                                            out_channels=64, 
                                            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
                                            stride = trial.suggest_categorical('strides', [1, 2])
                                            ),
                                    nn.ReLU(),
                                    nn.Dropout(
                                            p=trial.suggest_float('rate', 0.2, 0.6)
                                    ),
                                    nn.Conv1d(
                                            in_channels=64, 
                                            out_channels=64, 
                                            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
                                            stride = trial.suggest_categorical('strides', [1, 2])
                                    ),
                                    nn.ReLU(),
                                    nn.Dropout(
                                            p=trial.suggest_float('rate', 0.2, 0.6)
                                    ),
                                    nn.MaxPool1d(
                                    kernel_size=trial.suggest_categorical('kernel_size', [3, 5]), 
                                    stride = trial.suggest_categorical('strides', [1, 2]),
                                    ),
                                nn.AvgPool1d(
                                    kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
                                    stride = trial.suggest_categorical('strides', [1, 2]), 
                                    padding=0
                                    )
                                )
        #self.dropout1 = nn.Dropout(p=.2)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 
  
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.flat_fts = self.get_flat_fts((1, 187), self.features)

        self.classifier = nn.Sequential(
                                        nn.Linear(
                                                in_features=self.flat_fts, 
                                                out_features=64),
                                        nn.Linear(
                                                in_features=64, 
                                                out_features=5, 
                                                bias=True) 
        )

    def get_flat_fts(self, in_size, fts):
            f = fts(Variable(torch.ones(1, *in_size)))
            return int(np.prod(f.size()[1:]))


    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return F.log_softmax(out, dim=-1) 




#net = ecg_net()
#x = torch.randn(1, 1, 187)
#y = net(x)
#print(y.size())


