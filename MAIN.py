import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



IMGL_PATH = './images/c.jpeg'
IMGR_PATH = './images/d.jpeg'
WIN_SIZE = 7
DSR = 30
K = 16

class main():
    def __init__(self):
        self.img_l = cv.imread(IMGL_PATH,0)
        # self.img_l = np.resize(self.img_l,())
        self.img_r = cv.imread(IMGR_PATH,0)
        # self.img_l = cv.resize(self.img_l,(self.img_l.shape[1] // 4,self.img_l.shape[0] // 4),interpolation=cv.INTER_CUBIC)
        # self.img_r = cv.resize(self.img_r,(self.img_r.shape[1] // 4,self.img_r.shape[0] // 4),interpolation=cv.INTER_CUBIC)


    def subKernel(self,kernel_l,kernel_r):
        diff = np.abs(kernel_l - kernel_r)
        return np.sum(diff)





    def getDisparity(self,img_l,img_r):
        img_height = img_l.shape[0]
        img_width = img_l.shape[1]
        disparity = np.zeros((img_height,img_width))
        print(disparity.shape)
        for i in range(img_height - WIN_SIZE):
            for j in range(img_width - WIN_SIZE):
                left = img_l[i : i + WIN_SIZE,j : j + WIN_SIZE]

                cand = []
                for k in range(DSR):
                    y = j - k

                    if y >= 0:
                        right = img_r[i : i + WIN_SIZE,y: y + WIN_SIZE]
                        val = self.subKernel(left,right)
                        cand.append(val)
                loc = cand.index(min(cand))

                disparity[i,j] = np.int(loc * K)
        # np.savetxt('disparity',disparity)
        # cv.imshow("test",disparity)
        # cv.waitKey()
        return disparity


    def show(self,disparity):
        output = np.zeros([disparity.shape[0], disparity.shape[1], 3])

        for row in range(disparity.shape[0]):
            for col in range(disparity.shape[1]):
                color = disparity[row][col]
                if color <= 51:
                    output[row][col][0] = 255
                    output[row][col][1] = color * 5
                    continue

                if color <= 102:
                    color -= 51
                    output[row][col][0] = 255 - color * 5
                    output[row][col][1] = 255
                    continue

                if color <= 153:
                    color -= 102
                    output[row][col][1] = 255
                    output[row][col][2] = color * 5
                    continue

                if color <= 204:
                    color -= 153
                    output[row][col][1] = 255 - int(128.0 * color / 51.0 + 0.5)
                    output[row][col][2] = 255
                    continue

                else:
                    color -= 204
                    output[row][col][1] = 127 - int(128.0 * color / 51.0 + 0.5)
                    output[row][col][2] = 255

        cv.imshow("test", output)
        cv.waitKey()



    def mainMethod(self):
        disparity = self.getDisparity(self.img_l,self.img_r)
        self.show(disparity)




if __name__ == '__main__':
    t = main()
    t.mainMethod()
