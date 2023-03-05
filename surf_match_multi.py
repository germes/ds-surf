import cv2
from include import surf_lib
import os
import warnings
warnings.filterwarnings("ignore")

# Aim of this script is processing multiple images and compare the base image with
# transformed images - rotated, on changed scene, scaled, etc

# image_set = 'graffiti'
image_set = 'box'

input_path = 'images/' + image_set + '/'
output_path = 'output/' + image_set + '/'

file_list = os.listdir(input_path)
print(f'Found files: {file_list}')

matcher = surf_lib.SURF_Matcher()

print(f'Start processing base files')

img_base = cv2.imread(input_path + 'base.jpg')
gray_image_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)

surfObject = surf_lib.SURF(threshold=1000)
surfObject.init(gray_image_base)

keypoints_base, descriptors_base = surfObject.detectAndCompute()

# Print number of keypoints and descriptors
print(f'Number of keypoints Image Base: {len(keypoints_base)}')
print(f'Number of descriptors Image Base: {len(descriptors_base)}')

surf_lib.save_keypoints_to_file(keypoints_base, output_path + 'image_base_kp.txt')
surf_lib.save_descriptors_to_file(descriptors_base, output_path + 'image_base_des.txt')

kp_image = cv2.drawKeypoints(img_base, keypoints_base, None, color=(
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output_path + 'image_base_kp.jpg', kp_image)

print(f'Finish processing base file\n')

for file in file_list:
    if file == 'base.jpg':
        continue

    print(f'Start processing {file}')

    filename = os.path.splitext(file)[0]

    img = cv2.imread(input_path + file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surfObject_file = surf_lib.SURF(threshold=1000)
    surfObject_file.init(gray_image)

    keypoints, descriptors = surfObject_file.detectAndCompute()

    # Print number of keypoints and descriptors
    print(f'Number of keypoints Image {filename}: {len(keypoints)}')
    print(f'Number of descriptors Image {filename}: {len(descriptors)}')

    surf_lib.save_keypoints_to_file(keypoints_base, output_path + 'image_' + filename + '_kp.txt')
    surf_lib.save_descriptors_to_file(descriptors_base, output_path + 'image_' + filename + '_des.txt')

    kp_image = cv2.drawKeypoints(img, keypoints, None, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(output_path + 'image_' + filename + '_kp.jpg', kp_image)

    print(f'..keypoint printed, matching ..')

    matches = matcher.match(descriptors_base, descriptors)

    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 20 matches.
    img_match = cv2.drawMatches(img_base, keypoints_base, img, keypoints, matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite(output_path + 'image_' + filename + '_match.jpg', img_match)

    print(f'Finished processing {file}\n')

print('done!')
