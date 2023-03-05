import cv2
from include import surf_lib
import warnings
warnings.filterwarnings("ignore")

# Aim of this script is testing hwo SURF algorithm working
# as a result wll be provided image with keypoints

path_to_input_image = 'images/box/base.jpg'

img = cv2.imread(path_to_input_image)

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surfObject = surf_lib.SURF(threshold=1000)
surfObject.init(gray_image)

keypoints, _ = surfObject.detectAndCompute()
print(f'Number of keypoints: {len(keypoints)}')

kp_image = cv2.drawKeypoints(img, keypoints, None, color=(
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('output/output.jpg', kp_image)

print('done!')