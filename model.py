import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)
    - Freely choose activation functions as you want
    - For subsampling, use max pooling with kernel_size = (2,2)
    - Output should be a logit vector
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        img = self.relu(self.conv1(img))
        img = self.pool1(img)
        img = self.relu(self.conv2(img))
        img = self.pool2(img)
        img = img.view(img.size(0), -1)
        img = self.relu(self.fc1(img))
        img = self.relu(self.fc2(img))
        output = self.fc3(img)
        return output

class CustomMLP(nn.Module):
    """ Your custom MLP model
    
    - Note that the number of model parameters should be about the same with LeNet-5
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 90)
        self.fc2 = nn.Linear(90, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        img = img.view(img.size(0), -1)
        img = self.relu(self.fc1(img))
        img = self.relu(self.fc2(img))
        img = self.relu(self.fc3(img))
        output = self.fc4(img)
        return output