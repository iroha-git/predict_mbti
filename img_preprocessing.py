# 이미지 전처리
import cv2
import os
import numpy as np

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

    # cv2.imshow('facenet', img)  # 사진 출력
    # cv2.waitKey(10000)

cascade_filename = './haarcascade_frontalface_alt.xml'  # 가중치 파일 경로
cascade = cv2.CascadeClassifier(cascade_filename)  # 모델 불러오기

dataset_dir = './original_datasets'
mbti_lists = os.listdir(dataset_dir)

# preprocessed_datasets 폴더가 있는지 확인
if os.path.isdir('./preprocessed_datasets'):
    pass
else:
    os.mkdir('./preprocessed_datasets')

for mbti in mbti_lists:
    # mbti별 폴더가 있는지 확인
    if os.path.isdir(f'./preprocessed_datasets/{mbti}'):
        pass
    else:
        os.mkdir(f'./preprocessed_datasets/{mbti}')

    celeb_dir = f'./original_datasets/{mbti}'
    celeb_lists = os.listdir(celeb_dir)

    for celeb in celeb_lists:
        # 유명인별 폴더가 있는지 확인
        if os.path.isdir(f'./preprocessed_datasets/{mbti}/{celeb}'):
            pass
        else:
            os.mkdir(f'./preprocessed_datasets/{mbti}/{celeb}')

        img_dir = f'./original_datasets/{mbti}/{celeb}'
        img_lists = os.listdir(img_dir)

        for image in img_lists:
            img_array = np.fromfile(f'./original_datasets/{mbti}/{celeb}/{image}', np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 이미지 파일
            try:
                rtn_img = imgDetector(img, cascade)  # 사진 탐지기

                extension = os.path.splitext(image)[1]
                result, encoded_img = cv2.imencode(extension, rtn_img)
                new_image_path = f'./preprocessed_datasets/{mbti}/{celeb}/{image}'

                with open(new_image_path, 'wb') as f:
                    encoded_img.tofile(f)
            except:
                pass

    print(f'{mbti} | Complete!')
