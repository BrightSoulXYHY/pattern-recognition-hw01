import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D,self).__init__()
        # 160x20 -> 128x32 -> 64x32
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=32, kernel_size=33),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        # 64x32 -> 48x16 -> 24x16
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=17),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU()
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(24*16,48),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(48,8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8,2),
        )
   

    def forward(self,x):
        conv_layer1_out = self.conv_layer1(x)
        conv_layer2_out = self.conv_layer2(conv_layer1_out)
        out = self.fc_layer(conv_layer2_out)
        return out


class Attention_CNN(nn.Module):
    def __init__(self):
        super(Attention_CNN,self).__init__()
        self.att = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1,bias=0)
    
    def forward(self,x):
        out = self.att(x)
        return out
