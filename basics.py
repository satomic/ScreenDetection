# coding=utf-8

import cv2
from imutils.perspective import four_point_transform
from skimage.measure import compare_ssim as ssim
import numpy as np


def get_sharp_index(image, contour=None):
    '''
    # 用边缘所占的像素数目 除以 画面总的数目
    # 如果是指定某一个区域的话，那么只计算区域内的值
    :param image: image numpy array
    :param contour: contour if exist
    :return: sharp index
    '''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # init length of contour
    contour_length = 0

    # init area with full image size
    height, weight = gray.shape
    area = height * weight

    if contour is not None:
        # update paras with contour area
        area = cv2.contourArea(contour)
        contour_length = cv2.arcLength(contour, True)
        mask = np.zeros(gray.shape, np.uint8)
        # update grayed image with mask, only contour covered region is remain
        cv2.drawContours(mask, [contour], -1, (255), cv2.FILLED)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # sharp index in caculated by the edges/gray index
    edges = cv2.Canny(gray, 100, 200)
    edges_pixels_number = (edges > 1).sum() - contour_length
    sharp_index = 1.0 * edges_pixels_number / area
    return sharp_index


def is_clear(image, sharp_index=0.01, contour=None):
    return get_sharp_index(image,contour) > sharp_index


def get_max_rectangle_contour(image, mode=None, unpefect_ratio=0.1, area_ratio=0.02, debug=False):
    '''
    :param image: 图像，已经读取过的，彩色图像
    :param mode: 参考基准None表示灰度，0,1,2分别为BGR
    :param unpefect_ratio: 矩形的不完美比率最大
    :param area_ratio: 矩形面积占比整个图像的占比最小
    :param debug: 调试模式
    :return: 包含四个point的list
    '''
    if mode is None:
        # 转化为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image[:,:,mode]
        # cv2.imshow("gray",gray)

    # 计算当前图像的面积
    total_area = gray.shape[0] * gray.shape[1]
    if debug:
        print "image area:",total_area

    # 双边滤波，该滤波器可以在保证边界清晰的情况下有效的去掉噪声。
    # 它的构造比较复杂，即考虑了图像的空间关系，也考虑图像的灰度关系。
    # 双边滤波同时使用了空间高斯权重和灰度相似性高斯权重，确保了边界不会被模糊掉。
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.bilateralFilter(gray, 9, 45, 45)
    if debug:
        cv2.imshow("bilateralFiltered", gray)
    edged = cv2.Canny(gray, 30, 200)
    if debug:
        cv2.imshow("edged", edged)

    # 从边缘图中找出所有的等高线，找出其中面积最大的5个
    (img, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    if debug:
        cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)
    screenCnt = None

    # 从10个中判断出哪个才是屏幕
    for c in cnts:
        # 轮廓周长，第二个参数为True表示轮廓闭合
        peri = cv2.arcLength(c, True)

        # 轮廓近似
        # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，
        # 新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法，可以自己Google。
        # 假设我们要在一幅图像中查找一个矩形，但是由于图像的种种原因我们不能得到一个完美的矩形，
        # 而是一个“坏形状”，现在就可以使用这个函数来近似这个形状，第二个参数是epsilon，
        # 它是从原始轮廓到近似轮廓的最大距离，它是一个准确度参数。
        approx = cv2.approxPolyDP(c, unpefect_ratio * peri, True)

        # 经过模糊评估之后，如果用四个点可以近似，则认为是矩形，同时轮廓占比大于图像的一定比率
        if debug:
            print "current_area:",cv2.contourArea(approx)
        if len(approx) == 4 and cv2.contourArea(approx) > total_area * area_ratio:
            screenCnt = approx
            break
    return screenCnt


def get_region_by_contour(image, contour):
    mask = np.zeros(image.shape[0:2], np.uint8)
    # update grayed image with mask, only contour covered region is remain
    cv2.drawContours(mask, [contour], -1, (255), cv2.FILLED)
    return cv2.bitwise_and(image, image, mask=mask)


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if width and height:
        dim = (width, height)

    # check to see if the width is None
    elif width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_resized_region_by_contour(image, contour, height=800, width=480):
    region = four_point_transform(image, contour.reshape(4, 2))
    resized_region = resize(region, height=height, width=width)
    return resized_region


def get_normalized_3_color_distribution(image):
    # only support resized rectangle image
    if len(image.shape) != 3:
        raise Exception("the image is a gray scale!")
    B = image[:,:,0].sum()
    G = image[:,:,1].sum()
    R = image[:,:,2].sum()
    total = (B**2 + G**2 + R**2) ** 0.5
    return (B,G,R)/total


def BGR_shift_A_2_B(image_a, image_b):
    '''
    :param image_a: 
    :param image_b: 
    :return: color shift on channel B G R 
    '''
    color_dis_a = get_normalized_3_color_distribution(image_a)
    color_dis_b = get_normalized_3_color_distribution(image_b)
    return color_dis_b - color_dis_a


def compare_images_ssim(image_a, image_b, multichannel=True):
    return ssim(image_a, image_b, multichannel=multichannel)


def get_grayscale_spectrum(image):
    # return cv2.calcHist(image)
    pass


if __name__ == "__main__":
    img1 = cv2.imread("pic/phone_1.jpg")
    img2 = cv2.imread("pic/phone_1_compare.jpg")
    print compare_images_ssim(img1,img2,multichannel=True)

    k = cv2.waitKey(0)