## STEP 1: demarcation
we have the demarcation image
![avatar](https://raw.githubusercontent.com/satomic/ScreenDetection/master/pic/simu/1.jpg)

---

## STEP 2: get an image and resize it
the image is 
![avatar](https://raw.githubusercontent.com/satomic/ScreenDetection/master/pic/simu/3.jpg)
by resize the screen region, we compare it with the orignal one, we can say, it well enough
![avatar](https://raw.githubusercontent.com/satomic/ScreenDetection/master/pic/simu/resized_and_original.png)<br/>
the ssim value of the two images is `0.886242935676`, very similar.


---

## STEP 3 - 4: we repeat the step 2 to get another two images

---

## STEP 5: now we have 3 resized images, and all they have a BAD AREA, we can get it!
![avatar](https://raw.githubusercontent.com/satomic/ScreenDetection/master/pic/simu/bad_area.png)

---

# thanks to Aoi Miyazaki