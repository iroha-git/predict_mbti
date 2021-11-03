# 해당하는 인물의 사진이 이미 데이터셋에 존재하는가?
import os

data_dir = './datasets'

query = input("Enter name: ")

mbti_list = os.listdir(data_dir)
for mbti in mbti_list:
    path = os.path.join(data_dir, mbti)
    fnames = os.listdir(path)
    for fname in fnames:
        flag = query in fname
        if flag:
            print(f"Found {query} included file name!\nFile name = {fname}\nDir name = {mbti}")
            break
        else:
            pass
