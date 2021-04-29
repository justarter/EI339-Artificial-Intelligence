import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, data, transform


def translate(filename,i):
    # 读入图片并变成灰色
    image_path = "raw/" + str(i) + "/" + filename
    img_gray = io.imread(image_path)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    # 缩小到28*28
    translated_img = transform.resize(img_gray, (28, 28))
    translated_img = np.array(translated_img * 255, dtype=np.uint8)
    save_path = "processed/" + str(i) + "/" + str(i)+"-518030910283-"+os.path.splitext(filename)[0] + ".png"
    io.imsave(save_path, translated_img)
    return translated_img


def main(i):
    if not os.path.exists("processed"):
        print("No raw data")
        return
    if not os.path.exists("processed"):
        os.mkdir("processed")
    for (dirpath, dirnames, filenames) in os.walk("raw/"+str(i)):
        for filename in filenames:
            if os.path.splitext(filename)[1] in [".jpg", ".png"]:
                print("Converting %s"%filename)
                translate(filename, i)
    print("Transform finished")


if __name__ == "__main__":
    for i in range(1, 11):
        main(i)
