# coding=utf-8

from basics import *

cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    img = image.copy()
    screenCnt = get_max_rectangle_contour(image, unpefect_ratio=0.1, area_ratio=0.02, debug=False)

    cv2.imshow("phone screen", image)

    if screenCnt is not None:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        cv2.imshow("phone screen", image)
        resized_region = get_resized_region_by_contour(img, screenCnt)
        cv2.imshow("resized", resized_region)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()