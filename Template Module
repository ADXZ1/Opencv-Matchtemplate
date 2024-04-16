import cv2
import numpy as np
import os

path = "/home/zzz/workspace/src/ros_astra_camera/scripts"
os.chdir(path)

image_background_path = "./photos/Mario.jpg"
#多尺度用conins2 多模板用coins
image_search_path = "./photos/coins2.jpg"

class Template:

    def __init__(self):
        pass


    def find_best_results(self,image_background_path, image_search_path):

        
        image_background_rgb = cv2.imread(image_background_path)

        image_background = cv2.cvtColor(image_background_rgb,cv2.COLOR_BGR2GRAY)

        image_search = cv2.imread(image_search_path,0)

        res = self.findmatrix(image_background,image_search)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        h,w = image_search.shape[:2]

        crop_rect = cv2.rectangle(image_background_rgb,(max_loc[0]+w,max_loc[1]+h),(max_loc[0],max_loc[1]) ,(255,0,0),5)

        cv2.namedWindow('crop_rect',cv2.WINDOW_NORMAL)
        cv2.imshow("crop_rect",crop_rect)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_all_results(self,image_background_path, image_search_path):

        image_background_rgb = cv2.imread(image_background_path)

        image_background = cv2.cvtColor(image_background_rgb,cv2.COLOR_BGR2GRAY)

        image_search = cv2.imread(image_search_path,0)

        h,w = image_search.shape[:2]

        res = self.findmatrix(image_background,image_search)

        threshold = 0.8

        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(image_background_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)

        
        cv2.namedWindow('crop_rect',cv2.WINDOW_NORMAL)
        cv2.imshow("crop_rect",image_background_rgb)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_all_size_results(self,image_background_path, image_search_path,scales = None):
    
        image_background_rgb = cv2.imread(image_background_path)

        image_background = cv2.cvtColor(image_background_rgb,cv2.COLOR_BGR2GRAY)

        image_search = cv2.imread(image_search_path,0)

        h,w = image_search.shape[:2]

        if scales is None:
            scales = np.linspace(0.1, 1.0, 20)[::-1]  #从1.0到0.1，分为20组 [::-1] 这部分是将生成的数列反转，使得数列的顺序从1.0到0.1。

        for scale in scales:
            
            resized_image = cv2.resize(image_search, (int(w*scale), int(h*scale)))
            
            rescales_w,rescales_h = resized_image.shape[::-1]

            res = self.findmatrix(image_background,resized_image)

            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

            #设定阈值
            threshold = 0.8

            if(max_val > threshold):
                cv2.rectangle(image_background_rgb,(max_loc[0]+rescales_w,max_loc[1]+rescales_h),(max_loc[0],max_loc[1]),(0,0,255),2)
                cv2.imshow("crop_rect",image_background_rgb)
                cv2.waitKey(0)
                break
            
            cv2.destroyAllWindows()    

    def findmatrix(self,image_background,image_search):

        return cv2.matchTemplate(image_background,image_search,cv2.TM_CCOEFF_NORMED)




    # 模板匹配
if __name__ == '__main__':
    t = Template()
    t.find_best_results(image_background_path, image_search_path)
    t.find_all_results(image_background_path, image_search_path)
    t.find_all_size_results(image_background_path, image_search_path)
