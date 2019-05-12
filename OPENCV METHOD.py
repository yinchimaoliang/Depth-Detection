import numpy as np
import cv2
from matplotlib import pyplot as plt

IMGL_PATH = './images/c.jpeg'
IMGR_PATH = './images/d.jpeg'


imgL = cv2.imread(IMGL_PATH,0)
imgR = cv2.imread(IMGR_PATH,0)
print(imgL.shape)
# imgL = cv2.resize(imgL,(imgL.shape[0] // 4,imgL.shape[1] // 4),interpolation=cv2.INTER_CUBIC)
# imgR = cv2.resize(imgR,(imgR.shape[0] // 4,imgR.shape[1] // 4),interpolation=cv2.INTER_CUBIC)
# disparity range tuning
window_size = 3
min_disp = 0
num_disp = 320 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=8 * 3 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# print(imgL.shape)

def show(disparity):

    gray = np.array(disparity)
    output = np.zeros([gray.shape[0],gray.shape[1],3])
    print(disparity)
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            color = gray[row][col]
            if color <= 51:
                output[row][col][0] = 255
                output[row][col][1] = color * 5
                continue

            if color <= 102:
                color -= 51
                output[row][col][0] = 255 - color * 5
                output[row][col][1] = 255
                continue

            if gray[row][col] <= 153:
                color -= 102
                output[row][col][1] = 255
                output[row][col][2] = color * 5
                continue



            if gray[row][col] <= 204:
                color -= 153
                output[row][col][1] = 255 - int(128.0 * color / 51.0 + 0.5)
                output[row][col][2] = 255
                continue

            else:
                color -= 204
                output[row][col][1] = 127 - int(128.0 * color / 51.0 + 0.5)
                output[row][col][2] = 255
            # output[row][col][0] = abs(255 - color)
            # output[row][col][1] = abs(127 - color)
            # output[row][col][2] = abs(color)

    cv2.imshow("test",output)
    cv2.waitKey()

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
show(disparity)
# plt.imshow(disparity, 'gray')
# plt.show()
