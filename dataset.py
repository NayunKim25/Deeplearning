import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
    Args:
        data_dir (str): Directory path containing images.
        
    Note:
        - Each image should be preprocessed as follows:
          All values should be in a range of [0,1]
          Subtract mean of 0.1307, and divide by std 0.30.
          These preprocessing can be implemented using torchvision.transforms.
        - Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        # 이미지 경로 지정
        self.data_dir = data_dir
        
        # 이미지, 레이블 목록 생성
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        # 전처리 파이프라인 정의
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # 흑백 변환
            transforms.ToTensor(),  # 텐서로 변환
            transforms.Normalize(mean=0.1307, std=0.3081)  # All values should be in a range of [0,1], Subtract mean of 0.1307, and divide by std 0.30
        ])

    def __len__(self):
        # 데이터셋의 총 길이 반환
        return len(self.image_files)

    def __getitem__(self, idx):
        # 주어진 인덱스의 이미지와 레이블 반환
        file_name = self.image_files[idx]
        # 이미지 로드
        img_path = os.path.join(self.data_dir, file_name)
        img = Image.open(img_path)
        
        # 레이블 추출
        label = int(file_name.split('_')[1].split('.')[0])

        # 전처리
        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    # 이미지 저장된 디렉토리 경로
    data_dir = 'C:/Users/nayun/Downloads/mnist-classification/mnist-classification/data/train/train'  
    mnist_dataset = MNIST(data_dir)

    # 데이터셋의 첫 번째 이미지와 레이블을 출력
    img, label = mnist_dataset[0]
    print(f'first image label: {label}')
    print(f'image shape: {img.shape}')
