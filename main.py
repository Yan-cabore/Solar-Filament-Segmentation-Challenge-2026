from pathlib import Path
import cv2 as cv # type: ignore
import matplotlib.pyplot as plt # type: ignore

img_path = Path("Images/MAGFiLO_1.0_Kaggle_2026/test/test_images/20111114063134Lh.jpeg")


src = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

# IMAGE PRE-PROCESSING TO LOWER NOISE
gaussian = cv.GaussianBlur(src, ksize=(3,3), sigmaX=3)

# SOMEHOW, THIS FILTER HELPED ENHANCE THE FILAMENTS BORDERS
# I ORIGINALLY TRIED USING IT TO LOWER THE HIGH INTENSITY VALUES
median_3 = cv.medianBlur(gaussian, ksize=3)
median_5 = cv.medianBlur(gaussian, ksize=5)

# PERFORMING BLACK-HAT OPERATION TO ENHANCE THE FILAMENTS 
# IN THE IMAGE AND REMOVE THE WHITE SPACE
elipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21,21))

# I MADE TWO VERSIONS TO COMPARE THE MEDIAN BLUR
bhat_3 = cv.morphologyEx(median_3, cv.MORPH_BLACKHAT, elipse)
bhat_5 = cv.morphologyEx(median_5, cv.MORPH_BLACKHAT, elipse)


# BASIC IMAGE VIEWING
fig ,ax = plt.subplots(1,2, figsize=(12,12))

ax[0].imshow(bhat_3, cmap='gray')
ax[1].imshow(bhat_5, cmap='gray')

for a in ax:
    a.axis('off')

plt.show()



