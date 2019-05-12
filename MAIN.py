import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



IMGL_PATH = './images/c.jpeg'
IMGR_PATH = './images/d.jpeg'
WIN_SIZE = 7
DSR = 30

class main():
    def __init__(self):
        self.img_l = cv.imread(IMGL_PATH,0)
        self.img_r = cv.imread(IMGR_PATH,0)


    def subKernel(self,kernel_l,kernel_r):
        diff = np.abs(kernel_l - kernel_r)
        return np.sum(diff)





    def getDisparity(self,img_l,img_r):
        img_height = img_l.shape[0]
        img_width = img_l.shape[1]
        disparity = np.zeros((img_height,img_width))
        print(disparity.shape)
        for i in range(img_height - WIN_SIZE):
            for j in range(img_height - WIN_SIZE):
                left = img_l[i : i + WIN_SIZE,j : j + WIN_SIZE]

                cand = []
                for k in range(DSR):
                    y = j - k

                    if y >= 0:
                        right = img_r[i : i + WIN_SIZE,y: y + WIN_SIZE]
                        val = self.subKernel(left,right)
                        cand.append(val)
                loc = cand.index(min(cand))

                disparity[i,j] = loc * 16

        print(disparity)



        # for i in range(img_height - WIN_SIZE):
        #     for j in range(img_width - WIN_SIZE):
        #         left = img_l[i : i + WIN_SIZE,j : j + WIN_SIZE]
        #
        #         cand = []
        #         for k in range(DSR):
        #             right =






    def mainMethod(self):
        self.getDisparity(self.img_l,self.img_r)




if __name__ == '__main__':
    t = main()
    t.mainMethod()
