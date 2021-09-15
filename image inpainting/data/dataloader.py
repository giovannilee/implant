import torch
from torch.utils.data import Dataset
from PIL import Image
from os import listdir, walk
from os.path import join
from random import randint
from data.basicFunction import CheckImageFile, ImageTransform, MaskTransform
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GetData(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(GetData, self).__init__()

        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.masks = [join (dataRootK, files) for dataRootK, dn, filenames in walk(maskRoot) \
            for files in filenames if CheckImageFile(files)]
        self.numOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        
        img = Image.open(self.imageFiles[index])
        image_name = self.imageFiles[index].split("/")[-2] + "/" +  self.imageFiles[index].split("/")[-1]
        mask = Image.open("dataset/implant/mask_big/{}".format(image_name))
        ##########binary##############
        # img = np.asarray(img)
        # img = np.where(img>0, 100, 0)
        # img = Image.fromarray(img.astype(np.uint8))
        ##############################

        #mask.save("view/implant_view/ori_mask_{}".format(self.imageFiles[index].split("/")[-1]))
        groundTruth = self.ImgTrans(img.convert("RGB"))
        mask = self.maskTrans(mask.convert("RGB"))

        # we add this threshhold to force the input mask to be binary 0,1 values
        # the threshhold value can be changeble, i think 0.5 is ok
        threshhold = 0.5
        ones = mask >= threshhold
        zeros = mask < threshhold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # here, we suggest that the white values(ones) denotes the area to be inpainted, 
        # and dark values(zeros) is the values remained. 
        # Therefore, we do a reverse step let mask = 1 - mask, the input = groundTruth * mask, :).
        mask = 1 - mask
        inputImage = groundTruth * mask

        inputImage = torch.cat((inputImage, mask[0].view(1, 256, 256)), 0)

        # #inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)

        # groundTruth = groundTruth.numpy().transpose(1,2,0)
        # mask = mask.numpy().transpose(1,2,0)
        # inputImage = inputImage.numpy().transpose(1,2,0)
     
        # groundTruth = groundTruth * 255
        # mask = mask * 255
        # inputImage = inputImage * 255

  
        # cv2.imwrite("view/implant_view/gt_{}".format(self.imageFiles[index].split("/")[-1]), groundTruth )
        # cv2.imwrite("view/implant_view/mask_{}".format(self.imageFiles[index].split("/")[-1]), mask )
        # cv2.imwrite("view/implant_view/input_{}".format(self.imageFiles[index].split("/")[-1]), inputImage)
        # inputImage_ = inputImage_.numpy().transpose(1,2,0)
        # inputImage_ = inputImage_ * 255
        # cv2.imwrite("view/implant_view/_input_{}".format(self.imageFiles[index].split("/")[-1]), inputImage_)
        # exit()
        

        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)
