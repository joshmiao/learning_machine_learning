from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import random

"""
本程序用于生成模拟北京理工大学统一身份认证验证码，程序细节注释参考create_captcha.py
"""
class ValidCodeImg:
    def __init__(self, width=80, height=30, code_count=4, font_size=26, point_count=0, line_count=5,
                 font_dir="arialbi.ttf", img_format='png', is_show=False):
        self.width = width
        self.height = height
        self.code_count = code_count
        self.font_size = font_size
        self.point_count = point_count
        self.line_count = line_count
        self.img_format = img_format
        self.font_dir = './captcha_fonts/' + font_dir
        self.is_show = is_show


    @staticmethod
    def getRandomColor():
        c1 = random.randint(0, 120)
        c2 = random.randint(0, 120)
        c3 = random.randint(0, 120)
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
        image = Image.new('RGB', (self.width, self.height), (232, 236, 248))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_dir, size=self.font_size)

        temp = []
        for i in range(self.code_count):
            random_char = self.getRandomStr()
            draw.text((i * 20, -4), random_char, self.getRandomColor(), font=font)
            temp.append(random_char)
        valid_str = "".join(temp)

        for i in range(self.line_count):
            x1 = random.randint(0, self.width)
            x2 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            y2 = random.randint(0, self.height)
            draw.line((x1, y1, x2, y2), fill=self.getRandomColor())

        if self.is_show:
            image.show()
            print(valid_str)

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
    for i in range(1000):
        img = ValidCodeImg()
        data, valid_str = img.getValidCodeImg()
        f = open(path + valid_str.lower() + '.png', 'wb')
        f.write(data)