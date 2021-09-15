import os
import glob
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# input_ = cv2.imread("dataset/implant/teeth/train/cate4-00140_22.jpg")
# print(np.unique(input_))
# output = cv2.imread("./output/output.png")
# print(np.unique(output))
# exit()

# output = cv2.imread("output/output.png")
# print(np.unique(output))
# input_ = cv2.imread("output/input.png")
# print(np.unique(input_))

# exit()
# ori_img = cv2.imread("dataset/implant/teeth/train/cate4-00140_497.png")
# print(np.unique(ori_img))
# exit()
# test_imgs = glob.glob("dataset/implant/teeth/test/*.png")
# for te_img in test_imgs:
#     img_name = te_img.split("/")[-1]

#     input_ = cv2.imread("{}".format(te_img))

#     cv2.imwrite("dataset/implant/teeth/test_vis/{}".format(img_name), input_*5)

# img = cv2.imread("output/cate4-00138_25_input.png")
# input_ = cv2.resize(img, dsize=(600, 300), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("view/cate4-00138_27_output_re.png", input_)


# sinus_ian = glob.glob("dataset/implant/sinus_ian/train/*.png")
# for img in sinus_ian:
#     image = np.asarray(Image.open("{}".format(img)))
#     img_name = img.split("/")[-1].split(".")[0]
#     cv2.imwrite("dataset/implant/sinus_ian/dd.png", image*100)
#     exit()
    


# input_ = cv2.imread("output/cate4-00110_15_input.png")
# output_ = cv2.imread("output/cate4-00110_15_output.png")

# gen = output_ - input_
# print(gen.shape)
# cv2.imwrite("input.png", input_)
# cv2.imwrite("output.png", output_)
# cv2.imwrite("difference.png", gen)
# exit()


# original = cv2.imread("view/difference.png", cv2.IMREAD_GRAYSCALE)
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(original, kernel, iterations=1)
# dilation = cv2.dilate(original, kernel, iterations=2)
# cv2.imwrite("view/erode.png", erosion)
# cv2.imwrite("view/dilate.png", dilation)
# cv2.imwrite("view/ori.png", original)

# images = glob.glob("output/*_output.png")
# for image in tqdm(images):

#     img_name = image.split("/")[-1].split(".")[0].split("_")[0] + "_" + image.split("/")[-1].split(".")[0].split("_")[1]
#     output_ = cv2.imread("output/{}_output.png".format(img_name))
#     input_ = cv2.imread("output/{}_input.png".format(img_name))
#     gt_ = cv2.imread("output/{}_gt.png".format(img_name))
#     pred = output_ - input_
#     gt = gt_ - input_
#     cv2.imwrite("view/difference/{}_pred.png".format(img_name), pred)
#     cv2.imwrite("view/difference/{}_gt.png".format(img_name), gt)

preds = glob.glob("view/difference/*_pred.png")
for i in tqdm(preds):
    pred = cv2.imread("{}".format(i))
    img_name = i.split("/")[-1].split(".")[0]
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(pred, kernel, iterations=1)
    cv2.imwrite("view/difference/{}_erode.png".format(img_name), erosion)
 

