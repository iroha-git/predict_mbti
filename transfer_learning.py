# 딥러닝 모델 (전이학습)
import os
import torch
import torchvision.transforms as tf
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import time
import copy

def run():
    torch.multiprocessing.freeze_support()  # freeze_support() Error 해결
    batch_size = 256
    epoch = 30
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

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

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # 학습 데이터와 검증 데이터의 총 개수를 각각 저장

    class_names = image_datasets['train'].classes
    # 클래스 이름 목록 저장


    # Pre-Trained Model 불러오기
    resnet = models.resnet152(pretrained=True)
    num_features = resnet.fc.in_features     # FC Layer의 입력 채널 수
    resnet.fc = nn.Linear(num_features, 16)  # FC Layer을 교체
    resnet.to(device)

    criterion = nn.CrossEntropyLoss()        # 손실 함수 지정

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)
    # requires_grad=True로 설정된 Layer의 Parameter만 업데이트

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # Epoch에 따라 Learning Rate를 변경. 7 Epoch마다 0.1씩 곱해 Learning Rate를 감소시킴


    # Pre-Trained Model의 일부 Layer Freeze하기
    count = 0  # 해당 Layer가 몇번째 Layer인가?
    for child in resnet.children():  # children() 메서드: 모델의 자식모듈을 iterator로 반환
        count += 1
        if count < 6:  # Parameter를 업데이트하지 않을 상위 Layer들의 requires_grad 값을 False로 지정
            for param in child.parameters():
                param.requires_grad = False  # Parameter가 업데이트되지 않도록 설정


    # Transfer Learning 모델 학습과 검증을 위한 함수
    def train_resnet(model, criterion, optimizer, scheduler, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())  # 정확도가 가장 높은 모델을 저장
        best_acc = 0.0  # 가장 높은 정확도

        for epoch in range(num_epochs):
            print(f'--------------- epoch {epoch + 1} ---------------')  # 현재 진행중인 Epoch
            since = time.time()

            for phase in ['train', 'val']:  # 한 Epoch마다 Training 모드와 Validation 모드를 각각 실행
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0  # 모든 데이터의 Loss를 합산해 저장
                running_corrects = 0  # 올바르게 예측한 경우의 수

                for inputs, labels in dataloaders[phase]:  # 모델의 현재 모드에 해당하는 DataLoader에서 데이터를 입력받음
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):  # 학습 모드에서만 모델의 Gradient를 업데이트
                        outputs = model(inputs)  # output 값을 계산
                        _, preds = torch.max(outputs, 1)  # 16개의 클래스에 속할 각각의 확률값 중, 최대의 인덱스를 예측값으로 저장
                        loss = criterion(outputs, labels)  # 예측값과 Target값 사이의 Loss를 계산

                        if phase == 'train':
                            loss.backward()
                            # 위에서 계산한 Loss 값을 바탕으로 Back Propagation 시행, 각 Parameter에 계산한 Gradient 값을 할당
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)  # 하나의 배치에 대해 계산된 Loss 값에 데이터 수를 곱해 합산
                    running_corrects += torch.sum(preds == labels.data)  # 예측값과 Target이 같다면 증가
                if phase == 'train':
                    scheduler.step()  # 7 Epoch마다 LR을 다르게 조정
                    l_r = [x['lr'] for x in optimizer_ft.param_groups]  # Scheduler에 의해 LR이 조정되는 것을 확인하기 위함
                    print(f'Learning Rate: {l_r}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print(f'Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60}s')

        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)  # 정확도가 가장 높은 모델을 불러옴

        return model

    # 모델 학습 실행하기
    model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)  # Fine-Tuning
    torch.save(model_resnet50, 'resnet50.pt')  # 모델 저장

if __name__ == '__main__':
    run()
