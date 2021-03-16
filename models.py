import torch
from torch import Tensor, nn
import torch.nn.functional as F
import helpers


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden1_size, hidden2_size, dropout1, dropout2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.Dropout(dropout1),      
            nn.ReLU(),
            nn.BatchNorm1d(hidden1_size),
            nn.Linear(hidden1_size,hidden2_size),
            nn.Dropout(dropout2),            
            nn.ReLU(),
            nn.BatchNorm1d(hidden2_size),
            nn.Linear(hidden2_size,output_size),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class BaseNet(nn.Module):
    def __init__(self, channels=2, output_size=2):
        super(BaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(channels, 4, 2), 
            nn.Conv2d(4, 16, 2), 
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 3),       
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.classifier = Classifier(input_size=256, output_size=output_size, hidden1_size=170, hidden2_size=90, dropout1=0.30, dropout2=0.2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class SiameseNoShare(nn.Module):
    def __init__(self):
        super(SiameseNoShare, self).__init__()
        self.left_net = BaseNet(channels=1, output_size=10)
        self.right_net = BaseNet(channels=1, output_size=10)
        
        self.classifier = Classifier(input_size=20, output_size=2, hidden1_size=25, hidden2_size=10, dropout1=0.25, dropout2=0.2)

    def forward(self, x):
        l_x = x[:,0].unsqueeze(1)
        r_x = x[:,1].unsqueeze(1)
        
        l_x = self.left_net(l_x)
        r_x = self.right_net(r_x)

        cat = torch.cat((l_x, r_x), 1)      # concatenate two output
        x = self.classifier(cat)
        return x, l_x, r_x

# shared weights
class SiameseShare(nn.Module):
    def __init__(self):
        super(SiameseShare, self).__init__()
        self.net = BaseNet(channels=1, output_size=10)
        self.classifier = Classifier(input_size=20, output_size=2, hidden1_size=25, hidden2_size=10, dropout1=0.25, dropout2=0.2)

    def forward(self, x):
        x_tmp = {}
        for i in range(x.shape[1]):
            x_tmp[i] = x[:,i].unsqueeze(1)
            x_tmp[i] = self.net(x_tmp[i])
        
        cat = torch.cat((x_tmp[0], x_tmp[1]), 1)      # concatenate two output
        x_final = self.classifier(cat)
        return x_final, x_tmp[0], x_tmp[1]
