import torch.nn as nn
import torch.nn.functional as F

class FashionClassifierVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FashionClassifier(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, num_classes=10, hidden_layers=1, activation_fn=F.relu, dropout_prob=0.3, batch_norm=False):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(hidden_layers - 1)])

        for i in range(hidden_layers - 1):
            setattr(self, f'fc{i+1}', nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        setattr(self, f'fc{hidden_layers}', nn.Linear(hidden_size if hidden_layers > 1 else input_size, num_classes))
        
    def forward(self, x):
        for i in range(self.hidden_layers - 1):
            x = self.activation_fn(getattr(self, f'fc{i+1}')(x))
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.dropout(x)
        x = getattr(self, f'fc{self.hidden_layers}')(x)
        return x