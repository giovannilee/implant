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
import glob
from tqdm import tqdm
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


if not CheckImageFile(args.input):
    print('Input file is not image file!')
elif not CheckImageFile(args.mask):
    print('Input mask is not image file!')
elif args.pretrained == '':
    print('Provide pretrained model!')
else:
    img_paths = glob.glob("dataset/implant/test/teeth/*.png")
    count = 1
    psnr_total = 0
    ssim_total = 0
    for path in tqdm(img_paths):
        img_name = path.split("/")[-1].split(".")[0]
        image = Image.open(path)
        gt = Image.open(path)
        mask = Image.open("dataset/implant/test/mask_back/{}.png".format(img_name))

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
        cv2.imwrite("output/{}_gt.png".format(img_name), gt_)
        image_ = image.numpy().transpose(1,2,0)
        image_ = image_ * 255 *5
        cv2.imwrite("output/{}_input.png".format(img_name), image_)
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
        psnr_total += psnr.item()
        print("PSNR MEAN", psnr_total / count)
        # ssim = SSIM(gt.cpu().numpy(), output_.cpu().numpy()) 
        SSIM = pytorch_msssim.MSSSIM()
        gt = gt.unsqueeze(0)
        ssim = SSIM(gt.cpu(), output.cpu())
        ssim_total += ssim.item()
        print("SSIM MEAN", ssim_total / count)
        count += 1
        #####################

        save_image(output*5, "output/{}_output".format(img_name) + '.png')