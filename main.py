import dataset
from model import LeNet5, CustomMLP

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.train()  
    total_loss = 0
    correct = 0
    total = 0

    for data in trn_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    trn_loss = total_loss / len(trn_loader)
    acc = correct / total * 100  

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for data in tst_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    tst_loss = total_loss / len(tst_loader)
    acc = correct / total * 100  

    return tst_loss, acc

def plot_statistics(train_stats, test_stats, title):
    """ Plot training and testing statistics """
    epochs = np.arange(1, len(train_stats) + 1)
    train_losses, train_accuracies = zip(*train_stats)
    test_losses, test_accuracies = zip(*test_stats)

    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.show()

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 및 데이터 로더 생성
    trn_data = dataset.MNIST(data_dir='C:/Users/nayun/Downloads/mnist-classification/mnist-classification/data/train/train')
    tst_data = dataset.MNIST(data_dir='C:/Users/nayun/Downloads/mnist-classification/mnist-classification/data/test/test')

    trn_loader = DataLoader(trn_data, batch_size=64, shuffle=True)
    tst_loader = DataLoader(tst_data, batch_size=64, shuffle=False)

    # LeNet-5 모델 생성 및 초기화
    lenet_model = LeNet5().to(device)
    optimizer_lenet = optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Custom MLP 모델 생성 및 초기화
    mlp_model = CustomMLP().to(device)
    optimizer_mlp = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
    
    # 에폭 10으로 설정
    epochs = 10
    
    # LeNet-5 모델 훈련 및 테스트
    lenet_train_stats = []
    lenet_test_stats = []
    
    for epoch in range(epochs):
        trn_loss, trn_acc = train(lenet_model, trn_loader, device, criterion, optimizer_lenet)
        tst_loss, tst_acc = test(lenet_model, tst_loader, device, criterion)
        
        lenet_train_stats.append((trn_loss, trn_acc))
        lenet_test_stats.append((tst_loss, tst_acc))
        
        print(f'LeNet-5 | Epoch [{epoch+1}/{epochs}]: Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}% | Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%')
    
    # Custom MLP 모델 훈련 및 테스트
    mlp_train_stats = []
    mlp_test_stats = []
    
    for epoch in range(epochs):
        trn_loss, trn_acc = train(mlp_model, trn_loader, device, criterion, optimizer_mlp)
        tst_loss, tst_acc = test(mlp_model, tst_loader, device, criterion)
        
        mlp_train_stats.append((trn_loss, trn_acc))
        mlp_test_stats.append((tst_loss, tst_acc))
        
        print(f'Custom MLP | Epoch [{epoch+1}/{epochs}]: Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}% | Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%')
    
    plot_statistics(lenet_train_stats, lenet_test_stats, 'LeNet-5')
    plot_statistics(mlp_train_stats, mlp_test_stats, 'Custom MLP')

if __name__ == '__main__':
    main()


# Employ at least more than two regularization techniques to improve LeNet-5 model.

class LeNet5Improved(nn.Module):
    """ Improved LeNet-5 (LeCun et al., 1998)   
        - Includes Batch Normalization and Dropout for regularization
        - Activation function: ReLU
        - Subsampling: Max Pooling with kernel_size=(2, 2)
        - Output: Logit vector
    """
    
    def __init__(self):
        super(LeNet5Improved, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)  # Batch Normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)  # Batch Normalization
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1_dropout = nn.Dropout(p=0.5)  # Dropout
        self.fc2 = nn.Linear(120, 84)
        self.fc2_dropout = nn.Dropout(p=0.5)  # Dropout
        self.fc3 = nn.Linear(84, 10)  # MNIST has 10 classes
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, img):
        # Forward pass
        img = self.relu(self.conv1(img))
        img = self.bn1(img)  # Batch normalization
        img = self.pool1(img)
        img = self.relu(self.conv2(img))
        img = self.bn2(img)  # Batch normalization
        img = self.pool2(img)
        
       # 입력 이미지를 평면화
        img = img.view(img.size(0), -1)
        
        img = self.relu(self.fc1(img))
        img = self.fc1_dropout(img)  # Dropout
        img = self.relu(self.fc2(img))
        img = self.fc2_dropout(img)  # Dropout
        output = self.fc3(img)
        
        return output


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # GPU/CPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 및 데이터 로더 생성
    trn_data = dataset.MNIST(data_dir='C:/Users/nayun/Downloads/mnist-classification/mnist-classification/data/train/train')
    tst_data = dataset.MNIST(data_dir='C:/Users/nayun/Downloads/mnist-classification/mnist-classification/data/test/test')

    trn_loader = DataLoader(trn_data, batch_size=64, shuffle=True)
    tst_loader = DataLoader(tst_data, batch_size=64, shuffle=False)

    # 개선한 LeNet-5 모델 생성 및 초기화
    lenet_improved_model = LeNet5Improved().to(device)
    optimizer_lenet = optim.SGD(lenet_improved_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 에폭 10으로 설정
    epochs = 10
    
    # 개선한 LeNet-5 모델 훈련 및 테스트
    lenet_improved_train_stats = []
    lenet_improved_test_stats = []
    
    for epoch in range(epochs):
        trn_loss, trn_acc = train(lenet_improved_model, trn_loader, device, criterion, optimizer_lenet)
        tst_loss, tst_acc = test(lenet_improved_model, tst_loader, device, criterion)
        
        lenet_improved_train_stats.append((trn_loss, trn_acc))
        lenet_improved_test_stats.append((tst_loss, tst_acc))
        
        print(f'LeNet-5 Improved | Epoch [{epoch+1}/{epochs}]: Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}% | Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%')
    
    # 개선한 LeNet-5 모델의 plot 그리기
    plot_statistics(lenet_improved_train_stats, lenet_improved_test_stats, 'LeNet-5 Improved')

if __name__ == '__main__':
    main()
