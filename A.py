#!/usr/bin/python
#coding:utf-8
from captcha.image import ImageCaptcha
import random
import numpy as npy      # 数字图像处理
import matplotlib.pyplot as plt
from PIL import Image

# 验证码长度 可任意设置
captcha_length = 4
number = ['0','1','2','3','4','5','6','7','8','9']
letter = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
LETTER = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
char_set = number + letter + LETTER

# 生成验证码图片
def get_text_and_image():
	# 获得验证码文本，内容区分大小写，且验证码长度为4或6
	while 1:
		_text = []
		for i in range(captcha_length):
			ch = random.choice(char_set)
			_text.append(ch)
		if len(_text) == captcha_length:
			break

	_text = "".join(_text)  # 将各字符组成一个字符串
	img = ImageCaptcha()
	captcha = img.generate(_text) # 生成验证码

	_img = npy.array(Image.open(captcha)) #  矩阵化
	return _text , _img

if __name__ == '__main__':  # 导入时可不运行此代码

	i = 0
	for i in range(0,100):
		text , img = get_text_and_image()
		fig = plt.figure()
		# 添加标题查看正确验证码
		fig.suptitle(text, fontsize=16)
		# 添加子图查看正确验证码
		#ax = fig.add_subplot(111)
		#ax.text(0, -2,text)
		plt.imshow(img)
		# 去掉坐标轴
		plt.axis('off')
		str1 = "captcha_photo/" + str(i) + ".png"
		plt.savefig(str1)
		#plt.show()

