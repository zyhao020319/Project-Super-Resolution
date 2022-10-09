import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import preprocessing

from materials.datasets.BraTs2018.preprocessing import Preprocessing, STAGE


def create_grid(size, path):
    num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2), indexing='xy')

    plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path)  # 保存图像
    
    
if __name__ == "__main__":
    # create_grid((240, 240), path="results/grid.png")
    # image = np.zeros((100, 100), dtype=np.uint16)
    # cv2.imwrite("image.png", image)
    
    preprocessing = Preprocessing(min_nonzero_pixels=1024)
    preprocessing.save_preprocessed_images()