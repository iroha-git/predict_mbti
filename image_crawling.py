from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
from selenium import webdriver
import os
import time
import winsound

crawl_count = 50  # int(input('Enter Count of Crawl Images: '))

while True:
    keywords = input('Enter Keywords: ').split(',')
    while True:
        mbti = input('Enter Keywords\' mbti: ').upper()
        if mbti in ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']:
            break
        print('Unknown mbti!')
    for keyword in keywords:
        url = f'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={quote_plus(keyword)}'
        # 한글 검색 자동 변환
        driver = webdriver.Chrome('chromedriver')
        driver.get(url)
        time.sleep(3)
        path = f'C:/Users/IROHA/Desktop/SW프로젝트/관상 mbti 분류/original_datasets/{mbti}/{keyword}'
        txt_path = f'C:/Users/IROHA/Desktop/SW프로젝트/관상 mbti 분류/original_datasets_list/{mbti}.txt'

        if not os.path.isdir(path):
            os.mkdir(path)

        with open(txt_path, 'a') as f:
            f.write(keyword)
        f.close()

        html = driver.page_source
        soup = bs(html, 'html.parser')
        img = soup.select('div.thumb > a > img')

        num = 1
        for i in img:
            try:
                imgUrl = i['data-lazy-src']
                with urlopen(imgUrl) as f:
                    with open(f'{path}/' + str(num) + '.jpg', 'wb') as h:
                        # w - write b - binary
                        img = f.read()
                        h.write(img)
                num += 1
                if num > crawl_count:
                    break
            except:
                pass

        print(f'{keyword} Image Crawling has been completed.')
        driver.quit()

    winsound.MessageBeep(type=-1)
    sel = input('Execute again? (y/n): ')
    if sel.lower() != 'y':
        break