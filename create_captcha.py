from PIL import Image, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import os
import random


class ValidCodeImg:
    def __init__(self, width=200, height=50, code_count=4, font_size=24, point_count=10, line_count=2,
                 background_random=True, color_random=True, is_transform=True, is_filter=True,
                 font=None, img_format='png', is_show=False):

        """
        可以生成一个经过处理后的随机验证码的图片
        :param width: 图片宽度 单位px
        :param height: 图片高度 单位px
        :param code_count: 验证码个数
        :param font_size: 字体大小
        :param point_count: 噪点个数
        :param line_count: 划线个数
        :param background_random: 背景颜色是否随机（默认白色）
        :param color_random: 字符颜色是否随机（默认黑色）
        :param is_transform: 是否扭曲处理验证码图片
        :param is_filter: 是否滤镜处理验证码图片
        :param font: 验证码字体文件名（放置于'./captcha_fonts/'文件夹中）
        :param img_format: 图片格式
        :param is_show: 是否展示所生成的图片（供调试使用）
        :return 生成的图片的bytes类型的data以及字符串类型的验证码字符
        """

        self.width = width
        self.height = height
        self.code_count = code_count
        self.font_size = font_size
        self.point_count = point_count
        self.line_count = line_count
        self.img_format = img_format
        self.background_random = background_random
        self.color_random = color_random
        self.is_transform = is_transform
        self.is_filter = is_filter
        self.font_dir = './captcha_fonts/' + font
        self.is_show = is_show

    @staticmethod
    def getRandomColor():
        """获取随机RGB颜色"""
        c1 = random.randint(0, 255)
        c2 = random.randint(0, 255)
        c3 = random.randint(0, 255)
        return c1, c2, c3

    @staticmethod
    def getRandomStr():
        """获取一个随机字符串，包含大小写字母和数字"""
        random_num = str(random.randint(0, 9))
        random_low_alpha = chr(random.randint(97, 122))
        random_upper_alpha = chr(random.randint(65, 90))
        random_char = random.choice([random_num, random_low_alpha, random_upper_alpha])
        return random_char

    def getValidCodeImg(self):
        """根据所设置参数获取一个Image对象并返回得到的图片和验证码字符"""
        if self.background_random:  # 创建图片背景（颜色随机）
            image = Image.new('RGB', (self.width, self.height), self.getRandomColor())
        else:  # 创建图片背景（颜色默认白色）
            image = Image.new('RGB', (self.width, self.height), (255, 255, 255))

        draw = ImageDraw.Draw(image)  # 获取一个画笔对象，将图片对象传过去
        font = ImageFont.truetype(self.font_dir, size=self.font_size)  # 获取一个font字体对象

        temp = []
        for i in range(self.code_count):  # 循环个随机字符串
            random_char = self.getRandomStr()
            if self.color_random:  # 在图片上写入得到的随机字符串（颜色随机）
                draw.text((10 + i * random.randint(15, 20), random.randint(-4, 8)),  # 字符的纵向坐标和字符之间的间距随机
                          random_char, self.getRandomColor(), font=font)
            else:  # 在图片上写入得到的随机字符串（颜色默认白色）
                draw.text((10 + i * random.randint(15, 20), random.randint(-4, 8)),
                          random_char, (0, 0, 0), font=font)
            temp.append(random_char)  # 保存验证码串以返回
        valid_str = "".join(temp)

        # 划干扰线
        for i in range(self.line_count):
            x1 = random.randint(0, self.width)
            x2 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            y2 = random.randint(0, self.height)
            draw.line((x1, y1, x2, y2), fill=self.getRandomColor())

        # 画干扰点
        for i in range(self.point_count):
            draw.point([random.randint(0, self.width), random.randint(0, self.height)],
                       fill=self.getRandomColor())
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.arc((x, y, x + 4, y + 4), 0, 90, fill=self.getRandomColor())

        # 对验证码进行扭曲变换
        if self.is_transform:
            params = [1 - float(random.randint(1, 2)) / 100,
                      0,
                      0,
                      0,
                      1 - float(random.randint(1, 10)) / 100,
                      float(random.randint(1, 2)) / 500,
                      0.001,
                      float(random.randint(1, 2)) / 500
                      ]
            image = image.transform((self.width, self.height), Image.PERSPECTIVE, params)

        # 对验证码图像增加滤镜效果
        if self.is_filter:
            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # 展示生成的验证码
        if self.is_show:
            image.show()
            print(valid_str)

        # 保存生成的图片
        from io import BytesIO
        f = BytesIO()
        image.save(f, self.img_format)
        data = f.getvalue()
        f.close()

        return data, valid_str  # 返回得到的图片和验证码字符


if __name__ == '__main__':
    # path = "./captcha/"  # 生成训练用验证码图像的路径
    path = "./captcha_to_predict/"  # 生成检测用验证码图像的路径
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(1600):  # 设置验证码图片数量
        img = ValidCodeImg(width=random.randint(100, 100), height=random.randint(40, 40),  # 设置验证码宽和高为100像素
                           code_count=4, font_size=24,  # 验证码字符个数和字体大小
                           point_count=10, line_count=2,  # 验证码干扰点和线数目
                           is_transform=random.choice([True]),  # 是否添加扭曲效果
                           is_filter=random.choice([True]),  # 是否添加滤镜效果
                           background_random=random.choice([True]),  # 背景颜色是否随机
                           color_random=random.choice([True]),  # 字体颜色是否随机
                           font=random.choice(["ARLRDBD.TTF", "cambriab.ttf", "courbd.ttf",  # 验证码使用的字体
                                               "bahnschrift.ttf", "arial.ttf", "ariblk.ttf",
                                               "micross.ttf", "arialbi.ttf", "consolaz.ttf"]),
                           img_format='png', is_show=False)  # 选择验证码图片格式以及是否展示生成的图片
        data, valid_str = img.getValidCodeImg()  # 创建验证码图片以及对应字符串
        f = open(path + valid_str.lower() + '.png', 'wb')
        f.write(data)
