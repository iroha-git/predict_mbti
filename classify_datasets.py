# 데이터셋 분류
import os
import shutil
import math

original_dataset_dir = './preprocessed_datasets'
mbti_list = os.listdir(original_dataset_dir)

base_dir = './splitted'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

for usage in [train_dir, validation_dir, test_dir]:
    os.mkdir(usage)

for mbti in mbti_list:
    for usage in [train_dir, validation_dir, test_dir]:
        os.mkdir(os.path.join(usage, mbti))

# ---------- 여기까지 용도에 따른 디렉터리 생성 ----------

for mbti in mbti_list:
    celeb_path = os.path.join(original_dataset_dir, mbti)
    celebs = os.listdir(celeb_path)

    train_count = 0
    val_count = 0
    test_count = 0

    for celeb in celebs:
        path = f'./preprocessed_datasets/{mbti}/{celeb}'
        fnames = os.listdir(path)

        train_size = math.floor(len(fnames) * 0.6)
        validation_size = math.floor(len(fnames) * 0.2)
        test_size = math.floor(len(fnames) * 0.2)

        train_fnames = fnames[:train_size]
        for fname in train_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(train_dir, mbti), f'{train_count}.jpg')
            shutil.copyfile(src, dst)
            train_count += 1

        validation_fnames = fnames[train_size:(validation_size + train_size)]
        for fname in validation_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(validation_dir, mbti), f'{val_count}.jpg')
            shutil.copyfile(src, dst)
            val_count += 1

        test_fnames = fnames[(train_size + validation_size):(validation_size + train_size + test_size)]
        for fname in test_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(test_dir, mbti), f'{test_count}.jpg')
            shutil.copyfile(src, dst)
            test_count += 1
