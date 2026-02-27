from iris import IrisClassifier, get_iris_band
from filters import filters
import cv2 as cv

iris_classifier = IrisClassifier(filters)

img1 = cv.imread("image1.png")
img2 = cv.imread("image2.png")

iris1, mask1, _ = get_iris_band(img1)
iris2, mask2, _ = get_iris_band(img2)

score, _ = iris_classifier(iris1, iris2, mask1, mask2)

iris_code, mask_code, _ = iris_classifier.get_iris_code(iris1, mask1)

score, _ = iris_classifier.compare_iris_code_and_iris(iris2, iris_code, mask1, mask_code)
