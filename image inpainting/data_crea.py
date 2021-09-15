import os
import cv2
from PIL import Image, ImageDraw
import glob
import numpy as np
import random
from tqdm import tqdm
from statistics import mean, median

# def make_mask(missing_list):
#     mask_image = Image.new("L", (600,300), (0))
#     draw = ImageDraw.Draw(mask_image)
#     for mis_num in missing_list:
#         print(mis_num)
#         exit()
    



npys = glob.glob("./dataset/implant/teeth_npy/val_npy/*.npy")

mis_list = [12,13,14,15,16,17,18,
                           21,22,23,24,25,26,27,
                           32,33,34,35,36,37,38,
                           41,42,43,44,45,46,47]

for cnt in tqdm(range(300)):
    for npy in npys:
        file_name = npy.split("/")[-1].split(".")[0]

        tooth_list = []
        missing_list = []
        npy_info = np.load(npy, allow_pickle=True)
        for i in range(len(npy_info)):
            teeth_num = npy_info[i]["idx"]
            tooth_list.append(teeth_num)

        teeth_image = np.zeros((300,600))
        #mask_image = Image.new("L", (600,300), (0))
        #mask_image.save("./image.png")
        
        missing_button = True
        random.shuffle(mis_list)
        mis_teeth = random.choice(mis_list)


        for teeth in range(len(tooth_list)):
            file_name = npy.split("/")[-1].split(".")[0]
            teeth_num = npy_info[teeth]["idx"]
            rbox = npy_info[teeth]["rbox"]
            teeth_mask = npy_info[teeth]["mask"]
            inst_x = np.where(teeth_mask==1)[0]
            inst_y = np.where(teeth_mask==1)[1]

            if int(mis_teeth) == int(teeth_num):
                missing_button = False
                for k in range(len(inst_x)):
                    try:
                        teeth_image[inst_x[k]][inst_y[k]] = int(teeth_num)
                    except IndexError:
                        continue

            elif missing_button:
                
                for k in range(len(inst_x)):
                    try:
                        teeth_image[inst_x[k]][inst_y[k]] = int(teeth_num)
                    except IndexError:
                        continue 

            else:
                value = random.random()
                if value >= 0.5:
                    for k in range(len(inst_x)):
                        try:
                            teeth_image[inst_x[k]][inst_y[k]] = int(teeth_num)
                        except IndexError:
                            continue 
                else:
                    for k in range(len(inst_x)):
                        try:
                            teeth_image[inst_x[k]][inst_y[k]] = int(teeth_num)
                            
                        except IndexError:

                            continue 


                    # missing_list.append(teeth_num)
                    # mask_image1 = ImageDraw.Draw(mask_image)  
                    # mask_image1.rectangle([(min(inst_y), min(inst_x)), (max(inst_y), max(inst_x))], fill=(255), width=1) 
                     
                    # w = int((mean(inst_x) - min(inst_x)))
                    # h = int((mean(inst_y) - min(inst_y)))
                    
                    # #for y in range(w*2):
                    # for z in range(h*2):
                    #     mask_image[mean(inst_x)-w : mean(inst_x)+w][mean(inst_y)-h+z] = 255

        #mask_image = make_mask(missing_list)


        cv2.imwrite("dataset/implant/teeth_big/val/{}_{}.png".format(file_name, cnt), teeth_image)


        #mask_image.save("dataset/implant/mask_10/train/{}_{}.png".format(file_name, cnt))
       