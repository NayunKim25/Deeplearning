# 인공신경망과 딥러닝 HW2

MNIST Classification
: Build a neural network classifier with MNIST dataset

1. The number of model parameters of LeNet-5 and custom MLP.
  1-(1) LeNet-5
   - conv1: in_channels = 1, out_channels = 6, kernel_size = 5x5
     (1 * 5 * 5) * 6 + 6 = 150 + 6 = 156
   - conv2: in_channels = 6, out_channels = 16, kernel_size = 5x5
     (6 * 5 * 5) * 16 + 16 = 2,400 + 16 = 2,416
   - fc1: in_channels = 16 * 5 * 5 = 400, out_channels = 120
     400 * 120 + 120 = 48,000 + 120 = 48,120
   - fc2: in_channels = 120, out_channels = 84
     120 * 84 + 84 = 10,080 + 84 = 10,164
   - fc3: in_channels = 84, out_channels = 10
      84 * 10 + 10 = 840 + 10 = 850
     
   Total # of parameters = 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706

  1-(2) Custom MLP
   - fc1: in_channels = 28 * 28, out_channels = 90
      (28 * 28 + 1) * 90 = 70,590
   - fc2 = in_channels = 90, out_channels = 70
      (90 + 1) * 70 = 6,370
   - fc3 = in_channels = 70, out_channels = 50
      (70 + 1) * 50 = 3,550
   - fc4 = in_channels = 50, out_channels = 10
      (50 + 1) * 10 = 510

   Total # of parameters = 70,590 + 6,370 + 3,550 + 510 = 81,020

2. Plots for each model: loss and accuracy curves for training and test datasets.
  2-(1) LeNet-5
   LeNet-5 | Epoch [1/10]: Train Loss: 0.2811, Train Acc: 90.84% | Test Loss: 0.0614, Test Acc: 97.96%
   LeNet-5 | Epoch [2/10]: Train Loss: 0.0620, Train Acc: 98.03% | Test Loss: 0.0543, Test Acc: 98.28%
   LeNet-5 | Epoch [3/10]: Train Loss: 0.0431, Train Acc: 98.62% | Test Loss: 0.0449, Test Acc: 98.40%
   LeNet-5 | Epoch [4/10]: Train Loss: 0.0331, Train Acc: 98.99% | Test Loss: 0.0356, Test Acc: 98.86%
   LeNet-5 | Epoch [5/10]: Train Loss: 0.0271, Train Acc: 99.14% | Test Loss: 0.0402, Test Acc: 98.69%
   LeNet-5 | Epoch [6/10]: Train Loss: 0.0220, Train Acc: 99.32% | Test Loss: 0.0328, Test Acc: 98.92%
   LeNet-5 | Epoch [7/10]: Train Loss: 0.0181, Train Acc: 99.39% | Test Loss: 0.0377, Test Acc: 98.80%
   LeNet-5 | Epoch [8/10]: Train Loss: 0.0156, Train Acc: 99.50% | Test Loss: 0.0276, Test Acc: 99.11%
   LeNet-5 | Epoch [9/10]: Train Loss: 0.0128, Train Acc: 99.59% | Test Loss: 0.0289, Test Acc: 98.97%
   LeNet-5 | Epoch [10/10]: Train Loss: 0.0106, Train Acc: 99.66% | Test Loss: 0.0291, Test Acc: 99.09%

   ![image](https://github.com/NayunKim25/Deeplearning/assets/144984333/c9c3da91-0857-4d6c-9c89-f6f40d2bd0eb)

4. 
