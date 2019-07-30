'''
根据你要训练的验证码类型来设置如下参数
'''
import string

# 图像的通道数，32位图像为4，24位图像为3
CHANNELS = 4
# 图像的高度
HEIGHT = 50
# 图像的宽度
WIDTH = 130
# 验证码的长度
LEN = 5
# 验证码中的字符类别
__letters = string.digits + string.ascii_lowercase
CHARS = [c for c in __letters]
