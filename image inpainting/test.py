import os
import math
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets
from models.LBAMModel import LBAMModel
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from data.basicFunction import CheckImageFile
import cv2
import numpy as np
from psnr_ssim import *
import pytorch_msssim

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='input damaged image')
parser.add_argument('--mask', type=str, default='', help='input mask')
parser.add_argument('--output', type=str, default='output', help='output file name')
parser.add_argument('--pretrained', type=str, default='', help='load pretrained model')
parser.add_argument('--loadSize', type=int, default=256,
                    help='image loading size')
parser.add_argument('--cropSize', type=int, default=256,
                    help='image training size')
parser.add_argument('--gpu_id', type=str ,default='1')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

ImageTransform = Compose([
    Resize((args.cropSize,args.cropSize), interpolation=Image.NEAREST),
    ToTensor(),
])

MaskTransform = Compose([
    Resize((args.cropSize,args.cropSize), interpolation=Image.NEAREST),
    ToTensor(),
])


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

# def SSIM(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     print("img1", img1.shape)
#     print("img2", img2.shape)
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     print("ssim_map", ssim_map.shape)
#     exit()
#     return ssim_map.mean()


if not CheckImageFile(args.input):
    print('Input file is not image file!')
elif not CheckImageFile(args.mask):
    print('Input mask is not image file!')
elif args.pretrained == '':
    print('Provide pretrained model!')
else:
    
    image = Image.open(args.input)
    gt = Image.open(args.input)
    mask = Image.open(args.mask)
    

    #################binary################
    # image = np.asarray(image)
    # image = np.where(image>0, 1, 0)
    # image = Image.fromarray(image.astype(np.uint8))
    #######################################

    image = ImageTransform(image.convert('RGB'))
    gt = ImageTransform(gt.convert('RGB'))
    mask = MaskTransform(mask.convert('RGB'))

    threshhold = 0.5
    ones = mask >= threshhold
    zeros = mask < threshhold

    mask.masked_fill_(ones, 1.0)
    mask.masked_fill_(zeros, 0.0)

    mask = 1 - mask
    sizes = image.size()
    
    image = image * mask
    #########
    gt_ = gt.numpy().transpose(1,2,0)
    gt_ = gt_ * 255 *5
    cv2.imwrite("output/gt.png", gt_)
    image_ = image.numpy().transpose(1,2,0)
    image_ = image_ * 255 *5
    cv2.imwrite("output/input.png", image_)
    #########
    inputImage = torch.cat((image, mask[0].view(1, sizes[1], sizes[2])), 0)
    inputImage = inputImage.view(1, 4, sizes[1], sizes[2])

    
    mask = mask.view(1, sizes[0], sizes[1], sizes[2])
    
    netG = LBAMModel(4, 3)

    netG.load_state_dict(torch.load(args.pretrained))
    for param in netG.parameters():
        param.requires_grad = False
    netG.eval()
    print(netG.reverseConv5.updateMask.alpha)
    if torch.cuda.is_available():
        netG = netG.cuda()
        inputImage = inputImage.cuda()
        mask = mask.cuda()

    output = netG(inputImage, mask)
    output = output * (1 - mask) + inputImage[:, 0:3, :, :] * mask

    #######################
    output_ = output.squeeze(0)

    psnr = PSNR(gt.cpu().float(), output_.cpu().float())
    print("psnr", psnr)
    # ssim = SSIM(gt.cpu().numpy(), output_.cpu().numpy()) 
    ssim = pytorch_msssim.MSSSIM()
    gt = gt.unsqueeze(0)
    print("ssim", ssim(gt.cpu(), output.cpu()))
    #####################
    save_image(output*5, args.output + '.png')