from PIL import Image, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import os
import random


class ValidCodeImg:
    def __init__(self, width=200, height=50, code_count=5, font_size=32, point_count=20, line_count=3,
                 background_random=True, color_random=True, is_transform=True, is_filter=True,
                 font_dir=None, img_format='png', is_show=False):

        """
        可以生成一个经过降噪后的随机验证码的图片
        :param width: 图片宽度 单位px
        :param height: 图片高度 单位px
        :param code_count: 验证码个数
        :param font_size: 字体大小
        :param point_count: 噪点个数
        :param line_count: 划线个数
        :param img_format: 图片格式
        :return 生成的图片的bytes类型的data
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
        self.font_dir = './captcha_fonts/' + font_dir
        self.is_show = is_show


    @staticmethod
    def getRandomColor():
        c1 = random.randint(0, 255)
        c2 = random.randint(0, 255)
        c3 = random.randint(0, 255)
        return c1, c2, c3

    @staticmethod
    def getRandomStr():
        '''获取一个随机字符串，每个字符的颜色也是随机的'''
        random_num = str(random.randint(0, 9))
        random_low_alpha = chr(random.randint(97, 122))
        random_upper_alpha = chr(random.randint(65, 90))
        random_char = random.choice([random_num, random_low_alpha, random_upper_alpha])
        return random_char

    def getValidCodeImg(self):
        # 获取一个Image对象，参数分别是RGB模式。宽150，高30，随机颜色
        if self.background_random:
            image = Image.new('RGB', (self.width, self.height), self.getRandomColor())
        else:
            image = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        # 获取一个画笔对象，将图片对象传过去
        draw = ImageDraw.Draw(image)

        # 获取一个font字体对象参数是ttf的字体文件的目录，以及字体的大小
        font = ImageFont.truetype(self.font_dir, size=self.font_size)

        temp = []
        for i in range(self.code_count):
            # 循环5次，获取5个随机字符串
            random_char = self.getRandomStr()

            # 在图片上一次写入得到的随机字符串,参数是：定位，字符串，颜色，字体
            if self.color_random:
                draw.text((10 + i * random.randint(15, 20), random.randint(-4, 8)),
                          random_char, self.getRandomColor(), font=font)
            else:
                draw.text((10 + i * random.randint(15, 20), random.randint(-4, 8)),
                          random_char, (0, 0, 0), font=font)
            # 保存随机字符，以供验证用户输入的验证码是否正确时使用
            temp.append(random_char)
        valid_str = "".join(temp)

        # 噪点噪线
        # 划线
        for i in range(self.line_count):
            x1 = random.randint(0, self.width)
            x2 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            y2 = random.randint(0, self.height)
            draw.line((x1, y1, x2, y2), fill=self.getRandomColor())

        # 画点
        for i in range(self.point_count):
            draw.point([random.randint(0, self.width), random.randint(0, self.height)],
                       fill=self.getRandomColor())
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.arc((x, y, x + 4, y + 4), 0, 90, fill=self.getRandomColor())

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
            image = image.transform((self.width, self.height), Image.PERSPECTIVE, params)  # 扭曲
        if self.is_filter:
            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜
        if self.is_show:
            image.show()
            print(valid_str)
        # 在内存生成图片
        from io import BytesIO
        f = BytesIO()
        image.save(f, self.img_format)
        data = f.getvalue()
        f.close()

        return data, valid_str


if __name__ == '__main__':
    # path = "./captcha/"
    path = "./captcha_to_predict/"
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(1600):
        img = ValidCodeImg(width=random.randint(100, 100), height=random.randint(40, 40),
                           code_count=4, font_size=24,
                           point_count=10, line_count=2,
                           is_transform=random.choice([True]),
                           is_filter=random.choice([True]),
                           background_random=random.choice([True]),
                           color_random=random.choice([True]),
                           font_dir=random.choice(["ARLRDBD.TTF", "cambriab.ttf",
                                                   "courbd.ttf", "bahnschrift.ttf",
                                                   "arial.ttf", "ariblk.ttf",
                                                   "micross.ttf", "arialbi.ttf",
                                                   "consolaz.ttf"]),
                           img_format='png', is_show=False)
        data, valid_str = img.getValidCodeImg()
        f = open(path + valid_str.lower() + '.png', 'wb')
        f.write(data)