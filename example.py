# coding=utf-8

import cv2
from basics import *

# ==========================================================
# read and analysis orignal image
image = cv2.imread("pic/phone_1.jpg")
cv2.imshow("original image", image)
print "original image 3 color:", get_normalized_3_color_distribution(image)
print "original image sharp:",get_sharp_index(image)

# ==========================================================
# get max rectangele contour, and draw it on orignal image, then save it
cnt = get_max_rectangle_contour(image)
image_with_cnt = image.copy()
cv2.drawContours(image_with_cnt,[cnt],-1,(0,255,0),2)
cv2.imshow("image_with_cnt",image_with_cnt)
cv2.imwrite("example/image_with_cnt.jpg",image_with_cnt)

# ==========================================================
# get region by contour, with out resize, so there will be black outside the contour
region = get_region_by_contour(image,cnt)
cv2.imshow("region",region)
cv2.imwrite("example/region.jpg",region)
print "region sharp:", get_sharp_index(region)
print "region sharp with contour:", get_sharp_index(region,cnt)

# ==========================================================
# get resized region by contour, the return is en full display image
resized_region = get_resized_region_by_contour(image,cnt)
cv2.imshow("resized_region", resized_region)
cv2.imwrite("example/resized_region.jpg",resized_region)
print "resized_region 3 color:", get_normalized_3_color_distribution(resized_region)
print "resized_region sharp:", get_sharp_index(resized_region)

# ==========================================================
# how the color channel BGR shift from A to B
# positive channel means color increased, the bigger, the more increased
print BGR_shift_A_2_B(image, resized_region)

k = cv2.waitKey(0)