import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt



IMGL_PATH = './images/l1.png'
IMGR_PATH = './images/r1.png'
WIN_SIZE = 7
DSR = 30
K = 16
SCALAR = 1

class main():
    def __init__(self):
        self.img_l = cv.imread(IMGL_PATH,0)
        # self.img_l = np.resize(self.img_l,())
        self.img_r = cv.imread(IMGR_PATH,0)
        # self.img_l = cv.resize(self.img_l,(self.img_l.shape[1] // SCALAR,self.img_l.shape[0] // SCALAR),interpolation=cv.INTER_CUBIC)
        # self.img_r = cv.resize(self.img_r,(self.img_r.shape[1] // SCALAR,self.img_r.shape[0] // SCALAR),interpolation=cv.INTER_CUBIC)

        # print(self.img_r )
    def subKernel(self,kernel_l,kernel_r):
        sum_l = np.float64(np.sum(kernel_l))
        sum_r = np.float64(np.sum(kernel_r))
        # print(sum_l)
        # diff = np.abs(kernel_l - kernel_r)
        return np.abs(sum_l - sum_r)





    def getDisparity(self,img_l,img_r):
        img_height = img_l.shape[0]
        img_width = img_l.shape[1]
        disparity = copy.deepcopy(img_l)
        # disparity = np.zeros_like(img_l,dtype = 'uint8')
        # print(type(disparity[0][0]))
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
                disparity[i,j] = np.uint8(loc * K)
        # np.savetxt('disparity',disparity)
        # cv.imshow("test",disparity)
        # cv.waitKey()
        return disparity


    def show(self,disparity):
        output = np.zeros([disparity.shape[0], disparity.shape[1], 3],dtype = 'uint8')
        print(output.shape)
        # mask1 = np.where(disparity == 0)
        # output[mask1[0] , mask1[1]] = [0,0,50]
        #
        # mask2 = np.where((disparity < 0.1) & (disparity > 0))
        # output[mask2[0], mask2[1]] = [0, 0, 100]
        #
        # mask3 = np.where((disparity > 0.1) & (disparity < 0.2))
        # output[mask3[0],mask3[1]] = [0,0,150]
        #
        # mask4 = np.where((disparity > 0.2) & (disparity < 0.3))
        # output[mask4[0] , mask4[1]] = [0,0,200]
        #
        # mask5 = np.where((disparity > 0.3) & (disparity < 0.4))
        # output[mask5[0], mask5[1]] = [0, 0, 250]
        # output[mask1[0] , mask1[1] , 1] = 0
        # output[mask1[0] , mask1[1] , 2] = 50
        # print(output[a])
        for row in range(disparity.shape[0]):
            for col in range(disparity.shape[1]):
                color = disparity[row][col]
                if color <= 51:
                    output[row][col][0] = 0
                    output[row][col][1] = 0
                    output[row][col][2] = 0
                    continue

                if color <= 102:
                    output[row][col][0] = 0
                    output[row][col][1] = 0
                    output[row][col][2] = 128
                    continue

                if color <= 153:
                    output[row][col][0] = 0
                    output[row][col][1] = 128
                    output[row][col][2] = 0
                    continue

                if color <= 220:
                    output[row][col][0] = 128
                    output[row][col][1] = 0
                    output[row][col][2] = 0
                    continue

                else:
                    output[row][col][0] = 255
                    output[row][col][1] = 255
                    output[row][col][2] = 255

        cv.imshow("test", output)
        cv.waitKey()



    def mainMethod(self):
        disparity = self.getDisparity(self.img_l,self.img_r)
        # np.savetxt("dis",disparity)
        # print(disparity)
        # cv.imshow('test',disparity)
        # cv.waitKey()
        self.show(disparity)




if __name__ == '__main__':
    t = main()
    t.mainMethod()
