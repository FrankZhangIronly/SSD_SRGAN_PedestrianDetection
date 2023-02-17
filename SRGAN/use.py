from .utils import *
from torch import nn
import torch
from .models import SRResNet, Generator
import time
from PIL import Image
import cv2
import numpy as np

class My_SRGAN:
    def __init__(self):
        # 模型参数
        large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
        small_kernel_size = 3  # 中间层卷积的核大小
        n_channels = 64  # 中间层通道数
        n_blocks = 16  # 残差模块数量
        scaling_factor = 2  # 放大比例
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 预训练模型
        srgan_checkpoint = "./SRGAN/results/checkpoint_srgan_x2_full.pth"
        # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"
        # 加载模型SRResNet 或 SRGAN
        checkpoint = torch.load(srgan_checkpoint)
        generator = Generator(large_kernel_size=large_kernel_size,
                              small_kernel_size=small_kernel_size,
                              n_channels=n_channels,
                              n_blocks=n_blocks,
                              scaling_factor=scaling_factor)
        generator = generator.to(self.device)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        self.model = generator

    def use(self, cv2img):
        # cv2 to pil
        img = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB))
        img = img.convert('RGB')

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)

        # 转移数据至设备
        lr_img = lr_img.to(self.device)  # (1, 3, w, h ), imagenet-normed

        # 模型推理
        with torch.no_grad():
            sr_img = self.model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            img_np = cv2.cvtColor(np.asarray(sr_img), cv2.COLOR_RGB2BGR)
            # cv2.imshow('cv2image', img_np)
            # cv2.waitKey()
            # sr_img.save('./results/half_srgan_x4.png')
        return img_np

