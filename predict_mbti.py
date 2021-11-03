import os
import time
import copy
import torch
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

batch_size = 256
epoch = 30
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

class Net(nn.Module):
    def __init__(self):  # 모델에서 사용할 Layer 정의
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)  # 입력 채널, 출력 채널, 커널 크기
        self.pool = nn.MaxPool2d(2, 2)  # 커널 크기, Stride
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # 활성함수 ReLU
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 4096)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# 모델 학습
def train(model, train_loader, optimizer):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# 모델 평가
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def train_baseline(model, train_loader, val_loader, optimizer, num_epochs=30):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        since = time.time()
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'--------------- epoch {epoch} ---------------')

        print(f'train Loss: {train_loss:.4f} Acc: {train_acc:.2f}')
        print(f'val Loss: {val_loss:.4f} Acc: {val_acc:.2f}')
        print(f'Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60}s')

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

model_base = Net().to(device)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)

data_transforms = {
        'train': tf.Compose([  # 학습 데이터
            tf.Resize([64, 64]),           # 이미지의 크기 64*64
            tf.RandomHorizontalFlip(),     # 이미지를 무작위로 좌우 반전
            tf.RandomVerticalFlip(),       # 이미지를 무작위로 상하 반전
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Tensor 형태의 이미지 정규화
        ]),
        'val': tf.Compose([    # 검증 데이터
            tf.Resize([64, 64]),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

data_dir = './splitted'
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x),
                                 transform=data_transforms[x]) for x in ['train', 'val']}
# root: 데이터를 불러올 경로
# transform: 전처리 또는 Augmentation 방법 지정

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,  # 데이터를 배치 단위로 분리
                                              shuffle=True, num_workers=4) for x in ['train', 'val']}
# shuffle: 모델이 학습시 Label 순서를 기억하는 것을 방지

def run():
    torch.multiprocessing.freeze_support()  # freeze_support() Error 해결
    base = train_baseline(model_base, dataloaders['train'], dataloaders['val'], optimizer, num_epochs=epoch)

    torch.save(base, 'baseline.pt')

if __name__ == '__main__':
    run()

