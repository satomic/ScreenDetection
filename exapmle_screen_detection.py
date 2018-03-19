# coding=utf-8

import cv2
from basics import *


image_path = "pic/simu/"

"""
STEP 1: demarcation
"""
# load demarcation image
demarcation = cv2.imread(image_path + "1.jpg")
canny_demarcation = cv2.Canny(cv2.cvtColor(demarcation, cv2.COLOR_BGR2GRAY), 30, 200)
cv2.imshow("canny_demarcation", canny_demarcation)


# get the most important thing, the contour with screen region
contour = get_max_rectangle_contour(demarcation, mode=0, unpefect_ratio=0.1, area_ratio=0.02, debug=False)
region = get_region_by_contour(demarcation, contour)
# cv2.imshow("region", region)


"""
STEP 2: get image 3 and resize it
"""
# load image 3.jpg and resize it
image_3 = cv2.imread(image_path + "3.jpg")
resized_image_3 = get_resized_region_by_contour(image_3, contour, height=595, width=500)
cv2.imshow("resized_image_3", resized_image_3)

# compare it with the original.jpg, we can get the ssim = 0.886242935676, and obviously they are the same
image_original = cv2.imread(image_path + "original.jpg")
print compare_images_ssim(image_3, image_original, multichannel=True)


"""
STEP 3: get image 4 and resize it
"""
# load image 3.jpg and resize it
image_4 = cv2.imread(image_path + "4.jpg")
resized_image_4 = get_resized_region_by_contour(image_4, contour, height=595, width=500)


"""
STEP 4: get image 5 and resize it
"""
# load image 3.jpg and resize it
image_5 = cv2.imread(image_path + "5.jpg")
resized_image_5 = get_resized_region_by_contour(image_5, contour, height=595, width=500)


"""
STEP 4: now we have 3 resized images, and all they have a BAD AREA, we can get it!
"""
# get the canny of the 3  resized images
canny_3 = cv2.Canny(cv2.cvtColor(resized_image_3, cv2.COLOR_BGR2GRAY), 30, 200)
canny_4 = cv2.Canny(cv2.cvtColor(resized_image_4, cv2.COLOR_BGR2GRAY), 30, 200)
canny_5 = cv2.Canny(cv2.cvtColor(resized_image_5, cv2.COLOR_BGR2GRAY), 30, 200)

# dilate the edges
kernel = np.ones((3,3),np.uint8)
canny_3 = cv2.dilate(canny_3,kernel,iterations = 1)
canny_4 = cv2.dilate(canny_4,kernel,iterations = 1)
canny_5 = cv2.dilate(canny_5,kernel,iterations = 1)

cv2.imshow("canny_3", canny_3)
cv2.imshow("canny_4", canny_4)
cv2.imshow("canny_5", canny_5)

merged = cv2.bitwise_and(canny_3, canny_4)
merged = cv2.bitwise_and(merged, canny_5)

# clear the marginals
merged = merged[5:-5,5:-5]

# we can see the bad area on this clearly
cv2.imshow("merged", merged)

k = cv2.waitKey(0)