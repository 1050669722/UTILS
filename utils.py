import os
import cv2
import json
import numpy as np
import shutil
import colorsys
from PIL import Image, ImageDraw, ImageFont

def assignScale(ori_func):
    def new_func(name, img, scale=1.0):
        img = cv2.resize(img, (int(scale * img.shape[1]), int(scale * img.shape[0])))
        ori_func(name, img)
        return None

    return new_func


@assignScale
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def Array2Image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def Image2Array(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def getLabels(fpath="./train_annos.json"):
    """
    将json文件转换为字典
    :param fpath: json文件路径
    :return: 字典 #name -> {"image_height":..., "image_width":..., "category":..., "bbox":...} #一张图片的所有框及其类别
    """
    # 读取json
    with open(fpath, mode='r') as f:
        labels = json.load(f) #list
    # 应该把labels做成一个字典
    tmp = {}
    for label in labels:
        if label["name"] not in tmp:
            tmp[label["name"]] = {"image_height": label["image_height"], "image_width": label["image_width"], "categories": [label["category"]], "bboxes": [label["bbox"]]}
        else:
            tmp[label["name"]]["image_height"] = label["image_height"]
            tmp[label["name"]]["image_width"] = label["image_width"]
            tmp[label["name"]]["categories"].append(label["category"])
            tmp[label["name"]]["bboxes"].append(label["bbox"])
        # tmp[label["name"]] = {"image_height": label["image_height"], "image_width": label["image_width"], "category": label["category"], "bbox": label["bbox"], }
    # labels = tmp #fname -> {"image_height":..., "image_width":..., "category":..., "bbox":...}
    # del tmp
    # gc.collect()
    return tmp #labels #返回标签字典（格式化后的）


def becomesMultiple(m, n):
    """
    返回比m大的最小的n的整数倍值
    :param m: m
    :param n: n
    :return: 比m大的最小的n的整数倍值
    """
    return m + n - m % n if m % n != 0 else m #m + 1 * n - m % n


# def getLCM(m, n):
#     a, b = max(m, n), min(m, n)
#     while b:
#         temp = a % b
#         a = b
#         b = temp
#     c = m * n / a #最大公约数a #最小公倍数c
#     return int(c) #int(a), int(c)


def getfname(picpath):
    """
    通过 图片路径 获取 图片文件名
    :param picpath: 图片路径 或 图片名
    :return: 图片文件名
    """
    if os.sep in picpath:
        return picpath[picpath.rfind(os.sep) + 1:]
    elif '/' in picpath:
        return picpath[picpath.rfind('/') + 1:]
    else:
        return picpath


# def getfname2(picpath):
#     return picpath[picpath.rfind(os.sep) + 1:]


def splitGroups(List, num):
    """
    分组，用于多进程
    :param List: 待分组的列表
    :param num: 分组的数量
    :return: idxes 分好组的索引列表
    """
    assert isinstance(num, int)
    length = len(List)
    import math
    unitLength = math.ceil(length / num)
    idxes = []
    start = 0
    for i in range(0, length, unitLength):
        end = min(unitLength + i, length)
        idxes.append((start, end)) #每一批作为一个列表切片，放到一个res列表中
        start = end
    return idxes


def updatepath(path):
    # assert os.path.isdir(path), "Except {} be a path.".format(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return None


def PILBBox(pic, category_idx, bbox):
    """
    在指定数组格式的图片中，画框，写类别
    :param pic: 数组格式的图片
    :param category_idx: 类别索引 #从1开始
    :param bbox: 框坐标 #x1, y1, x2, y2
    :return: 画完框，写好类别的，数组格式的图片
    """
    # 缺陷类别
    defectKinds = ["boaderDefect", "cornerDefect", "whiteSpot", "grayBlobDefect", "deepSpot", "irisDefect", "marker", "scratch"]

    # 颜色
    hsv_tuples = [(x / len(defectKinds), 1., 1.) for x in range(len(defectKinds))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # 类别名字
    category = "{}".format(defectKinds[category_idx - 1])  # .encode("utf-8") #categories中的类别序号是从1开始的

    # 坐标
    left, top, right, bottom = [round(coor) for coor in bbox]  # 这里使用了round，而非int

    # 画框 #写类别
    image = Array2Image(pic)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='./simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = (np.shape(image)[0] + np.shape(image)[1]) // np.shape(image)[0]
    label_size = draw.textsize(category, font)
    text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array(
        [left, top + 1])  # 默认将标签写在框的上方，如果字体过大则将标签写在框内部
    for t in range(thickness):
        draw.rectangle([left + t, top + t, right - t, bottom - t],
                       outline=colors[category_idx - 1])  # [左上角x，左上角y，右下角x，右下角y], outline边框颜色
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[category_idx - 1])
    draw.text(text_origin, str(category.encode("utf-8"), "utf-8"), fill=(0, 0, 0), font=font)
    del draw

    # 输出数组格式的图片
    return Image2Array(image)


if __name__ == "__main__":
    print(becomesMultiple(6000, 4))
