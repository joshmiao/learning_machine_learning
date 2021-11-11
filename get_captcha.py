import requests

url = "http://login.bit.edu.cn/authserver/getCaptcha.htl?"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
}

for i in range(16):
    img = requests.get(url)
    with open(str(i) + ".png", "wb") as f:
        f.write(img.content)
