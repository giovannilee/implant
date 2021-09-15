import os
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm


teeths = glob.glob("dataset/implant/teeth_big/val/*.png")


mis_list = [12,13,14,15,16,17,18,
                           21,22,23,24,25,26,27,
                           32,33,34,35,36,37,38,
                           41,42,43,44,45,46,47]

for teeth in tqdm(teeths):
    ran_num = random.randrange(1,10)
    missing_list = random.sample(mis_list,ran_num)
    img_name = teeth.split("/")[-1]
    img = cv2.imread("dataset/implant/teeth_big/val/{}".format(img_name))

    for mis_num in missing_list:    
        img = np.where(img==mis_num, 255, img)
    
    img = np.where(img==0, 255, img)
    img = np.where(img!=255, 0, img)
    cv2.imwrite("dataset/implant/mask_big/val/{}".format(img_name), img)
    