**# 인공신경망과 딥러닝 HW2**

MNIST Classification
: Build a neural network classifier with MNIST dataset

**1. The number of model parameters of LeNet-5 and custom MLP.**
   
  **1-(1) LeNet-5**
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
     
   **Total # of parameters** = 156 + 2,416 + 48,120 + 10,164 + 850 = **61,706**

  **1-(2) Custom MLP**
   - fc1: in_channels = 28 * 28, out_channels = 90
      (28 * 28 + 1) * 90 = 70,590
   - fc2 = in_channels = 90, out_channels = 70
      (90 + 1) * 70 = 6,370
   - fc3 = in_channels = 70, out_channels = 50
      (70 + 1) * 50 = 3,550
   - fc4 = in_channels = 50, out_channels = 10
      (50 + 1) * 10 = 510

   **Total # of parameters** = 70,590 + 6,370 + 3,550 + 510 = **81,020**

**2. Plots for each model: loss and accuracy curves for training and test datasets.**
   
  **2-(1) LeNet-5**
  
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

  **2-(2) Custom MLP**
  
   Custom MLP | Epoch [1/10]: Train Loss: 0.3950, Train Acc: 87.68% | Test Loss: 0.1262, Test Acc: 95.95%
   
   Custom MLP | Epoch [2/10]: Train Loss: 0.1102, Train Acc: 96.62% | Test Loss: 0.0935, Test Acc: 97.10%
   
   Custom MLP | Epoch [3/10]: Train Loss: 0.0722, Train Acc: 97.83% | Test Loss: 0.0800, Test Acc: 97.42%
   
   Custom MLP | Epoch [4/10]: Train Loss: 0.0518, Train Acc: 98.41% | Test Loss: 0.0706, Test Acc: 97.78%
   
   Custom MLP | Epoch [5/10]: Train Loss: 0.0382, Train Acc: 98.85% | Test Loss: 0.0797, Test Acc: 97.44%
   
   Custom MLP | Epoch [6/10]: Train Loss: 0.0289, Train Acc: 99.05% | Test Loss: 0.0757, Test Acc: 97.68%
   
   Custom MLP | Epoch [7/10]: Train Loss: 0.0221, Train Acc: 99.27% | Test Loss: 0.0837, Test Acc: 97.70%
   
   Custom MLP | Epoch [8/10]: Train Loss: 0.0153, Train Acc: 99.53% | Test Loss: 0.0669, Test Acc: 98.06%
   
   Custom MLP | Epoch [9/10]: Train Loss: 0.0135, Train Acc: 99.56% | Test Loss: 0.0799, Test Acc: 97.98%
   
   Custom MLP | Epoch [10/10]: Train Loss: 0.0084, Train Acc: 99.75% | Test Loss: 0.0721, Test Acc: 98.16%
   
   ![image](https://github.com/NayunKim25/Deeplearning/assets/144984333/91ddc117-9ddd-4f59-b32e-a0cf97a6e8ba)

**3. Compare the predictive performances of LeNet-5 and custom MLP.**

   2의 결과를 보면 Train Loss는 LeNet-5 모델이 0.2811에서 0.0106으로 감소하였고, Custom MLP 모델은 0.3950에서 0.0084로 감소하였다. 
   따라서 최종 Train Loss는 Custom MLP가 더 낮았으며, Train Accuracy는 LeNet-5 모델이 90.84%에서 99.66%, 
   Custom MLP 모델이 87.68%에서 99.75%로 향상하여 Custom MLP가 높았다.
   또한, Test Loss는 LeNet-5 모델이 0.0614에서 010271로 감소하였고, Custom MLP 모델은 0.1262에서 0.0721로 감소하였다. 
   따라서 최종 Test Loss는 LeNet-5가 더 낮았으며, Test Accuracy는 LeNet-5 모델이 97.96%에서 99.09%로 향상되었으며
   Custom MLP 모델은 95.95%에서 98.16%로 향상하여 LeNet-5가 전반적으로 높았다.

   이를 통해 Train sets에 대해서는 Custom MLP가 약간 더 좋은 성능을 보였으나, Test sets에 대해서는 LeNet-5가 더 우수한 성능을 보였으므로
   일반화 성능을 고려하면 LeNet-5 모델의 예측 성능이 뛰어날 것이라고 예상된다. 
   또한, 이 정확도는 일반적으로 알려진 정확도와 유사한 것으로 보여진다.

**4. Employ at least more than two regularization techniques to improve LeNet-5 model.**

  Batch Normalization, Dropout의 두가지 방법을 사용하여 LeNet-5 모델을 개선하였다.
  그 결과 다음과 같은 결과를 얻을 수 있었다.
  
  LeNet-5 Improved | Epoch [1/10]: Train Loss: 0.3275, Train Acc: 89.80% | Test Loss: 0.0566, Test Acc: 98.28%
  
  LeNet-5 Improved | Epoch [2/10]: Train Loss: 0.1084, Train Acc: 97.14% | Test Loss: 0.0409, Test Acc: 98.77%
  
  LeNet-5 Improved | Epoch [3/10]: Train Loss: 0.0813, Train Acc: 97.96% | Test Loss: 0.0413, Test Acc: 98.82%
  
  LeNet-5 Improved | Epoch [4/10]: Train Loss: 0.0708, Train Acc: 98.13% | Test Loss: 0.0369, Test Acc: 98.82%
  
  LeNet-5 Improved | Epoch [5/10]: Train Loss: 0.0603, Train Acc: 98.41% | Test Loss: 0.0364, Test Acc: 98.98%
  
  LeNet-5 Improved | Epoch [6/10]: Train Loss: 0.0559, Train Acc: 98.43% | Test Loss: 0.0321, Test Acc: 99.11%
  
  LeNet-5 Improved | Epoch [7/10]: Train Loss: 0.0513, Train Acc: 98.62% | Test Loss: 0.0316, Test Acc: 99.11%
  
  LeNet-5 Improved | Epoch [8/10]: Train Loss: 0.0437, Train Acc: 98.78% | Test Loss: 0.0320, Test Acc: 99.07%
  
  LeNet-5 Improved | Epoch [9/10]: Train Loss: 0.0389, Train Acc: 98.92% | Test Loss: 0.0342, Test Acc: 99.16%
  
  LeNet-5 Improved | Epoch [10/10]: Train Loss: 0.0381, Train Acc: 98.92% | Test Loss: 0.0316, Test Acc: 99.17%
  
![image](https://github.com/NayunKim25/Deeplearning/assets/144984333/f7faef5e-740c-4e78-9683-8ee468588c80)

  2-(1)과 비교하여 최종 Test Accuracy가 99.17%로 조금 더 높은 향상된 것을 확인할 수 있었다.


- reference
https://pytorch.org/docs/stable/index.html

https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html

https://refstop.github.io/uda-lenet.html

https://resultofeffort.tistory.com/103

https://deep-learning-study.tistory.com/368

https://www.youtube.com/watch?v=otJfhQDytd0
