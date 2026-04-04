# %% 1. importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %% 1-5. 그래픽카드(GPU) 확인Python: Select Interpreter 및 설정
# PyTorch에게 "GPU가 있으면 쓰고, 없으면 CPU를 써라!" 하고 지시하는 아주 중요한 녀석이야.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("====================================")
print(f"현재 할당된 장치: {device}")
if device.type == 'cuda':
    # 내 컴퓨터에 꽂힌 그래픽카드 이름이 뭔지 자랑스럽게 출력해 줘!
    print(f"그래픽 카드 이름: {torch.cuda.get_device_name(0)}")
print("====================================")

# %% 2. loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# %% 3. exploring the dataset
train_features, train_labels = next(iter(trainloader))
print("The shape of the training features is:", train_features.shape)
print("The shape of the training labels is:", train_labels.shape)

# %% 4.
test_features, test_labels = next(iter(testloader))
print("The shape of the testing features is:", test_features.shape)
print("The shape of the testing labels is:", test_labels.shape)

# %% 5. visualizing some samples from the dataset
img = train_features[5]

# print()를 써줘야 화면에 잘 나와. 잘못 들어갔던 출력값은 지웠어!
print("이미지 형태:", img.shape) 

plt.figure(figsize=(4, 4))
plt.imshow(img.squeeze(0), 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

# 라벨값도 깔끔하게 숫자만 뽑아오려면 .item()을 붙여주면 좋아.
print("5번째 라벨 값:", train_labels[5].item())

# %% 6. 데이터를 그래픽카드(GPU)로 올리기 (사용 예시)
# 나중에 신경망 모델(Model)을 만들면 모델도 .to(device)로 올리고, 
# 데이터도 이렇게 .to(device)로 올려야 그래픽카드가 쌩쌩 돌아가면서 계산을 해!
train_features_gpu = train_features.to(device)
train_labels_gpu = train_labels.to(device)

print(f"데이터가 있는 곳: {train_features_gpu.device}")