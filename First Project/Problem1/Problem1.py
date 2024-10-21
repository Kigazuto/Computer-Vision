import cv2  # 读取图像和图像的保存
import numpy as np
import matplotlib.pyplot as plt

def load_img_pix(file_path):
    """获取图片对应的灰度像素值"""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图像的格式读取图像文件
    cv2.imshow('Before Equalization', img)
    cv2.waitKey(0)  # 键盘事件等待时间
    return img

def get_gray_histogram(img, height, width):
    """获取图像的灰度直方图"""
    gray = np.zeros(256)  # 保存各个灰度级（0-255）的出现次数

    for h in range(height):
        for w in range(width):
            gray[img[h][w]] += 1
    # 将直方图归一化, 即使用频率表示直方图
    gray /= (height * width)  # 保存灰度的出现频率，即直方图
    return gray

def plot_histogram(y, name):
    """绘制直方图
		- x:表示0-255的灰度值
		- y：表示这些灰度值出现的频率
	"""
    plt.figure(num=name)
    x = np.arange(0, 256)
    plt.bar(x, y, width=1)
    plt.show()

def get_gray_cumulative_prop(gray):
    """获取图像的累积分布直方图，即就P{X<=x}的概率
		- 大X表示随机变量
		- 小x表示取值边界
	"""
    cum_gray = []
    sum_prop = 0.
    for i in gray:
        sum_prop += i
        cum_gray.append(sum_prop)  # 累计概率求和
    return cum_gray

def pix_fill(img, cum_gray, height, width):
    """像素填充"""
    des_img = np.zeros((height, width), dtype=int)  # 定义目标图像矩阵

    for h in range(height):
        for w in range(width):
            des_img[h][w] = int(cum_gray[img[h][w]] * 255.0 + 0.5)

    cv2.imwrite("F:\Pycharm\Computer Vision\Problem1\Sherlock.jpg", des_img)  # 图像保存，返回值为bool类型
    des_img = cv2.imread("F:\Pycharm\Computer Vision\Problem1\Sherlock.jpg")
    return des_img

def show_pic(img, info="img"):
    """显示cv2图像对象"""
    cv2.imshow(chinese_encoder(info), img)
    cv2.waitKey(0)

def run_histogram_equalization(file_path):
    """图像均衡化执行函数"""
    img = load_img_pix(file_path)  # 获取图像的像素矩阵
    height, width = img.shape  # 返回图像的高和宽
    gray = get_gray_histogram(img, height, width)  # 获取图像的直方图
    cum_gray = get_gray_cumulative_prop(gray)  # 获取图像的累积直方图
    des_img = pix_fill(img, cum_gray, height, width)  # 根据均衡化函数（累积直方图）将像素映射到新图片
    plot_histogram(gray, "均衡化前的直方图")
    plot_histogram(cum_gray, "均衡化的累积函数")
    new_gray = get_gray_histogram(des_img, height, width)  # 获取图像均衡化后的直方图
    plot_histogram(new_gray, "均衡化后的直方图")
    show_pic(des_img, "After Equalization")

def chinese_encoder(string):
    return string.encode("gbk").decode(errors="ignore")

if __name__ == '__main__':
    file_path = r'F:\Pycharm\Computer Vision\Problem1\Sherlock.jpg'
    run_histogram_equalization(file_path)