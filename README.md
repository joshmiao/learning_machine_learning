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
>|requests|2.26.0|[下载Tensorflow==2.26.0](https://pypi.tuna.tsinghua.edu.cn/packages/e7/01/3569e0b535fb2e4a6c384bdbed00c55b9d78b5084e0fb7f4d0bf523d7670/requests-2.26.0.tar.gz)|


## 3. 总体框架
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
