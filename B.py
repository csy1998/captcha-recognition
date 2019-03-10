from A import get_text_and_image
from A import captcha_length
from A import char_set
from A import number  # 48-57
from A import letter  # 97-122
from A import LETTER  # 65-90
import numpy as npy
import tensorflow as tf

IMG_HEIGHT = 60
IMG_WIDTH = 160
char_set_len = len(char_set)
#text , img = get_text_and_image()
#char_set = number + letter + LETTER + ['_'] # '_'补齐验证码
print("欢迎打开验证码自动识别程序")
print("验证码字符长度：" , captcha_length)

# part 1 基础函数
##################################################################################

# 彩色转灰度图像，便于识别 RGB 转 黑白
def color2gray(img):
	if len(img.shape) <= 2:
		return img
	else:
		# 计算矩阵行均值 并返回一列矩阵
		gray_img = npy.mean(img , -1)
		return gray_img


# text 转 向量
def text2vector(text):
	#if len(text) > captcha_length:
	#	raise ValueError('验证码长度错误，超过要求最大长度')

	# 将字符转换成在字符集中的位置，函数内定义函数
	def char2ord(ch):
		p = ord(ch) - 48
		if p >= 0 and p < 10:
			return p
		p = ord(ch) - 65
		if p >= 0 and p < 26:
			return p + 10
		p = ord(ch) - 97
		if p >= 0 and p < 26:
			return p + 36
		return ValueError('CHAR_to_ASCII Error...')

	# 向量初始化为 0 并且大小为验证码长度 * 字体集大小
	vector = npy.zeros(captcha_length * char_set_len)
	# 每个字符填充进向量中
	for i , ch in enumerate(text):
		x = i * char_set_len + char2ord(ch)
		vector[x] = 1
	return vector


# 向量 转 text
def vector2text(vector):
	# char_nonzero 为向量全部非零元素组成的集合
	char_nonzero = vector.nonzero()[0]
	text = []
	for i , ch in enumerate(char_nonzero):
		char_x = ch % char_set_len
		if char_x < 10:
			char_p = char_x + ord('0')
		elif char_x < 36:
			char_p = char_x - 10 + ord('A')
		elif char_x < 62:
			char_p = char_x - 36 + ord('a')
		else:
			raise ValueError('Invalid Char...')
		text.append(chr(char_p))
	# 字符合成
	return "".join(text)


# 生成训练数据样本
def get_train_batch(batch_size = 128):
	# set_x 每行代表一张图片
	# set_y 每行代表一个验证码字符文本
	set_x = npy.zeros([batch_size , IMG_HEIGHT * IMG_WIDTH])
	set_y = npy.zeros([batch_size , captcha_length * char_set_len])

	# 防止生成错误验证码
	def get_right_captcha():
		while True:
			_text , _img = get_text_and_image()
			if _img.shape == (60 , 160 , 3):
				return _text , _img

	for i in range(batch_size):
		text , img = get_right_captcha()
		img = color2gray(img)  # 转成灰度图像
		# flatten 将矩阵折叠成灰度一维数组
		set_x[i,:] = img.flatten() / 255   # 值在0～1之间
		set_y[i,:] = text2vector(text)

	return  set_x ,set_y

#################################################################################################
# part 1 基础函数结束


# part 2 CNN算法部分
#################################################################################################

# tf储存临时替换变量
# sess.run()使用feed_dict={}传入 tf.placeholder() 变量
X = tf.placeholder(tf.float32 , [None , IMG_HEIGHT * IMG_WIDTH])
Y = tf.placeholder(tf.float32 , [None , captcha_length * char_set_len])
keep_prob = tf.placeholder(tf.float32)

# CNN 算法函数 卷积神经网络
def captcha_cnn(w_alpha = 0.01 , b_alpha = 0.1):

	# ############################################################################
	# 卷积神经网络各层功能概述
	# 卷积层：二维的卷积核作为过滤器扫描筛选出图片特征
	# 激励层：给卷积层中刚经过线性计算操作（元素相乘再求和）的系统引入非线性特征
	# 池化层：最大池化或平均池化 一能减少权重参数的数目从而降低计算成本。二能控制过拟合
	# Dropout层：丢弃该层中一个随机激活参数集，即在前向通过中将此激活参数集设为0 缓解过拟合问题
	# 	     注：只能在训练中使用，而不能用于测试过程
	# ############################################################################
	#
	# ############################################################################
	# tf 相关函数用法
	# tf.random_normal() 生成随机多维矩阵
	# tf.nn.relu(features, name = None) 计算激活函数relu，即max(features, 0)
	# tf.nn.bias_add(value, bias, name = None） 将偏差项 bias 加到 value 上面
	# tf.sigmoid(x, name = None) 计算x的sigmoid函数 计算公式为 y = 1 / (1 + exp(-x))
	# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
	# 		对一个四维的输入数据 input 和四维的卷积核 filter 进行操作，
	# 		然后对输入数据进行一个二维的卷积操作，最后得到卷积之后的结果。
	# 		输入张量的维度是 [batch, in_height, in_width, in_channels]
	# 		卷积核张量的维度是 [filter_height, filter_width, in_channels, out_channels]
	# 		必须有 strides[0] = strides[3] = 1
	# 		卷积核的水平移动步数和垂直移动步数是相同的，即 strides = [1, stride, stride, 1]
	# 		池化操作时步长stride = 2
	# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
	# 		计算神经网络层的dropout 概率keep_prob决定是否神经元放电
	# tf.nn.max_pool(value, ksize, strides, padding, name=None) 计算池化区域中元素的最大值
	#
	# #############################################################################


	# 调整形状，生成张量 -1表缺省，系统自动计算batch数
	x = tf.reshape(X , shape=[-1 , IMG_HEIGHT , IMG_WIDTH , 1])



	# 3层卷积网络
	# 生成随机卷积核w_c1和随机偏置b_c1
	w_c1 = tf.Variable(w_alpha * tf.random_normal([3,3,1,32]))
	b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
	# 生成第一卷积层：卷积操作 + 添加偏置 + 激励 + 最大池化 + dropout层处理过拟合
	# strides为移动步长 padding = SAME 向下取舍 ksize为矩阵窗口大小
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	# 生成第二卷积层
	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	# 生成第三卷积层
	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# 全链接层
	w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	# 输出层
	w_out = tf.Variable(w_alpha*tf.random_normal([1024, captcha_length * char_set_len]))
	b_out = tf.Variable(b_alpha*tf.random_normal([captcha_length * char_set_len]))
	# 构建线性模型
	output = tf.add(tf.matmul(dense, w_out), b_out)
	return output


# 训练函数 训练模型并保存
def train_captcha_cnn():

	# ##########################################################################
	# tf 相关函数用法
	# tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
	# 		分类函数 计算 logits 经 sigmoid 函数激活之后的交叉熵 度量概率误差
	# tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
	# 		分类函数 计算 logits 经 softmax 函数激活之后的交叉熵 度量概率误差
	# tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
	# 		求平均值 axis不设置则为求整体平均值
	# tf.train.AdamOptimizer(learning_rate=0.001, beta1, beta2, epsilon, use_locking, name=’Adam’)
	# 		实现了Adam算法的优化器 一个寻找全局最优点的优化算法，引入了二次方梯度校正
	# tf.argmax(input, axis=None, name=None, dimension=None)
	# 		返回最大值索引 axis=2 即是计算char_set_len维度中最大值索引
	# tf.equal(A, B) 对比两矩阵相等元素，相等就返回True，否则返回False，返回矩阵维度和A相同
	#
	#
	# ###########################################################################

	output = captcha_cnn()
	# 最小化误差 ; optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, targets=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	# 进行测试
	predict = tf.reshape(output, [-1, captcha_length , char_set_len])
	# 取出预测和损失的最大值索引
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, captcha_length , char_set_len]), 2)
	# 取出相同元素索引
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	# 强制类型转换之后求平均值得到准确率
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		# sess初始化所有变量
		sess.run(tf.initialize_all_variables())
		step = 0
		while True:
			set_x, set_y = get_train_batch(64)
			op_ , loss_ = sess.run([optimizer, loss], feed_dict={X: set_x, Y: set_y, keep_prob: 0.75})
			print("step:" , step , "    loss:" , loss_)

			# 每100步用模型进行测试计算一次准确率
			if step % 100 == 0:
				# 随机生成100个测试样本
				set_x_test, set_y_test = get_train_batch(100)
				acc = sess.run(accuracy, feed_dict={X: set_x_test, Y: set_y_test, keep_prob: 1.})
				print("当前步数:" , step, "    准确率:" , acc)
				# 如果准确率大于 res 则保存模型 并结束训练
				res = 0.5
				if acc > res and step > 50000:
					# 模型保存的路径及名称，名称后会加上当前保存模型的训练步数
					saver.save(sess, "训练模型/capcha.model", global_step=step)
					break
			step += 1

#######################################################################################
# part 2 CNN算法部分


# part 3 测试部分
# #######################################################################################
#  多次测试函数
def test_captcha():

	output = captcha_cnn()
	# create a saver 用于测试
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# 测试阶段恢复变量
		# 从特定目录下取出最近一次保存的模型恢复到sess
		saver.restore(sess , tf.train.latest_checkpoint("训练模型"))

		suc = 0
		fail = 0
		R = 1000
		for i in range(0 , R):
			c_text , c_img = get_text_and_image()
			c_img = color2gray(c_img)
			c_img = c_img.flatten() / 255
			predict = tf.argmax(tf.reshape(output , [-1,captcha_length,char_set_len]) , 2)
			text_list = sess.run(predict , feed_dict={X: [c_img] , keep_prob:1})
			# 转成01一维矩阵 便于转成向量从而得到测试后结果
			text = text_list[0].tolist()
			vector = npy.zeros(captcha_length * char_set_len )
			i = 0
			for j in text:
				vector[i*char_set_len + j] = 1
				i += 1
			predict_text = vector2text(vector)
			print( "正确为: {}  识别为: {}".format(c_text , predict_text))
			if c_text == predict_text:
				suc += 1
				print("------------------------------成功!!!!!")
			else:
				fail += 1
				print("------------------------------失败!")
			#return predict_text
		print("成功:  " , suc , " 失败:  " , fail)


# 单次测试
def one_test():
	# 转灰度矩阵 + 一维化 + 调整范围0～1
	text , img = get_text_and_image()
	img = color2gray(img)
	img = img.flatten() / 255
	predict_text = test_captcha(text , img);
	print( "正确验证码: {} 识别验证码: {}".format(text , predict_text))
	if text == predict_text:
		print("成功！")
	else:
		print("失败！")

# ######################################################################################
# part 3 测试部分



# 主程序开始
# ######################################################################################
if __name__ == '__main__':

	# 先训练 若只训练请注销test_captcha()
	#train_captcha_cnn()

	# 再测试 请注销train_captcha_cnn()
	 test_captcha()




