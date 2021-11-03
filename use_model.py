import torch
import cv2
import torchvision.transforms as tf
from torchvision.datasets import ImageFolder

model_path = './transfer_learning_17_2.pt'
model = torch.load(model_path)
device = torch.device('cuda')
model.to(device)

# 사진 출력
def imgDetector(img, cascade):
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)  # 사진 압축
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # 그레이 스케일 변환

    # cascade 얼굴 탐지 알고리즘
    results = cascade.detectMultiScale(gray,              # 입력 이미지
                                       scaleFactor=1.1,   # 이미지 피라미드 스케일 factor
                                       minNeighbors=5,    # 인접 객체 최소 거리 픽셀
                                       minSize=(20, 20))  # 탐지 객체 최소 크기

    for box in results:  # 결과값 = 탐지된 객체의 경계상자 list
        x, y, w, h = box  # 좌표 추출
        cropped_img = img[y: y+h, x: x+w]
        return cropped_img
        # img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)  # 경계 상자 그리기


cascade_filename = './haarcascade_frontalface_alt.xml'  # 가중치 파일 경로
cascade = cv2.CascadeClassifier(cascade_filename)  # 모델 불러오기

img = cv2.imread('./test_imgs/test8.jpg')
rtn_img = imgDetector(img, cascade)  # 사진 탐지기

convert_option = [
    tf.Resize([64, 64]),           # 이미지의 크기 64*64
    tf.ToTensor()
]

preprocessed_imgs = list()  # 결과가 여러개일 수도 있으므로 리스트로 생성

try:
    if rtn_img.any():
        for result in rtn_img:
            for option in convert_option:
                result = option(result)  # 입력 이미지 전처리

            preprocessed_imgs.append(result)

    else:
        print('얼굴을 찾을 수가 없어요!')
except:
    print('얼굴을 찾을 수가 없어요!')

