# 利用Tensorflow机器学习框架进行验证码识别
## 1. 项目简介

## 2. 系统环境

### Python解释器版本
>Python 3.9.7 (tags/v3.9.7:1016ef3, Aug 30 2021, 20:19:38) [MSC v.1929 64 bit (AMD64)] on win32
### 第三方库
>|库名称|版本|下载地址
>|-----|----|----|
>|Tensorflow|2.6.0|[下载Tensorflow==2.6.0](https://pypi.tuna.tsinghua.edu.cn/packages/fb/93/d5e3751a9ca3d159cbe498ef112e4bca35a07cedaae83e61038606e72edf/tensorflow-2.6.0-cp39-cp39-win_amd64.whl)|
>|numpy|1.19.5|[下载numpy==1.19.5](https://pypi.tuna.tsinghua.edu.cn/packages/bc/40/d6f7ba9ce5406b578e538325828ea43849a3dfd8db63d1147a257d19c8d1/numpy-1.19.5-cp39-cp39-win_amd64.whl)|
>|Pillow|8.4.0|[下载Pillow==8.4.0](https://pypi.tuna.tsinghua.edu.cn/packages/20/ec/15a263f2c65d71cf62aa767f774c2381077e07beb1e9309a94461ec1cd29/Pillow-8.4.0-cp39-cp39-win_amd64.whl)|
>|matplotlib|3.4.3|[下载matplotlib==3.4.3](https://pypi.tuna.tsinghua.edu.cn/packages/59/ea/1c00d9278c51d5f03276ac3f08773a13d93cbf2d722386ae8da083866697/matplotlib-3.4.3-cp39-cp39-win_amd64.whl)|
>|requests|2.26.0|[下载requests==2.26.0](https://pypi.tuna.tsinghua.edu.cn/packages/e7/01/3569e0b535fb2e4a6c384bdbed00c55b9d78b5084e0fb7f4d0bf523d7670/requests-2.26.0.tar.gz)|


## 3. 项目文件总体框架
```
│  create_captcha.py-----------创建训练图片
│  create_model.py-------------创建模型
│  dir.txt---------------------目录框架
│  get_captcha.py--------------爬虫下载验证码（北理网站）
│  predict.py------------------预测验证码
│  README.md-------------------说明文件
│  train_model.py--------------训练模型
│  
├─captcha_fonts----------------验证码字体文件集合
│      arial.ttf
│      arialbi.ttf
│      ariblk.ttf
│      ARLRDBD.TTF
│      bahnschrift.ttf
│      cambriab.ttf
│      consolaz.ttf
│      courbd.ttf
│      micross.ttf
│      msyh.ttc
│      
├─model------------------------模型数据（不可用于预测，可载入模型继续训练）
│      checkpoint
│      model_weights.data-00000-of-00001
│      model_weights.index
│      
└─model_to_predict-------------完整已训练的模型（可用于预测）
    │  keras_metadata.pb
    │  saved_model.pb
    │  
    ├─assets
    └─variables
            variables.data-00000-of-00001
            variables.index
```
## 4. 关键代码说明
### 创建训练用的验证码图片
```python
img = ValidCodeImg(width=random.randint(100, 100), height=random.randint(40, 40),# 设置验证码宽和高为100像素
                           code_count=4, font_size=24, # 验证码字符个数和字体大小
                           point_count=10, line_count=2, # 验证码干扰点和线数目
                           is_transform=random.choice([True]),# 是否添加扭曲效果
                           is_filter=random.choice([True]),# 是否添加滤镜效果
                           background_random=random.choice([True]),# 背景颜色是否随机
                           color_random=random.choice([True]),# 字体颜色是否随机
                           font_dir=random.choice(["ARLRDBD.TTF", "cambriab.ttf", # 验证码使用的字体
                                                   "courbd.ttf", "bahnschrift.ttf",
                                                   "arial.ttf", "ariblk.ttf",
                                                   "micross.ttf", "arialbi.ttf",
                                                   "consolaz.ttf"]),
                           img_format='png', is_show=False) # 选择验证码图片格式以及是否展示生成的图片
data, valid_str = img.getValidCodeImg() # 创建验证码图片以及对应字符串
```
### 输入数据预处理
```python
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
```
### 建立模型
```python

```
### 利用模型进行预测
```python

```
### 利用”北理统一身份认证“验证码测试模型在未训练的数据集上的准确度
```python
import requests

url = "http://login.bit.edu.cn/authserver/getCaptcha.htl?" # 验证码生成url
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
}
for i in range(16):
    img = requests.get(url) # 获取验证码
    with open(str(i) + ".png", "wb") as f:
        f.write(img.content) # 将验证码写入文件保存
```