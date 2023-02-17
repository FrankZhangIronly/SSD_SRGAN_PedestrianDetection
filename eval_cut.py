from __future__ import print_function
import os
import argparse
import cv2
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.augmentations import py_nms, draw_rec
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

from SRGAN.use import My_SRGAN

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_VOC_93000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/new_full/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='./data/VOCdevkit/', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

srgan = My_SRGAN()

def cut_srgan(img):
    width, hight, depth = np.shape(img)
    cutw, cuth, cutkw, cutkh= int(width/4), int(hight/4), int(width/2), int(hight/2)
    img_cut = img[cutw:cutw+cutkw, cuth:cuth+cutkh]
    img_cut_srgan = srgan.use(img_cut)
    # cv2.imshow('cv2image',img)
    # cv2.waitKey()
    # cv2.imshow('cv2image',img_cut)
    # cv2.waitKey()
    # cv2.imshow('cv2image',img_cut_srgan)
    # cv2.waitKey()
    return img_cut_srgan

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'eval.txt'
    num_images = len(testset)
    #num_images = 30
    if (os.path.exists(filename)):
        os.remove(filename)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img_org = testset.pull_image(index)
        img_id, annotation = testset.pull_anno(index)
        coords_list = []
        with open(filename, mode='a', encoding="gbk") as f:
            #f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            f.write('Annotations/'+img_id[:-4] + '\t')
        for ifcut in [False, True]:
            if ifcut == True:
                img = cut_srgan(img_org)
            else:
                img = img_org
            x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))

            if cuda:
                x = x.cuda()

            y = net(x)      # forward pass
            detections = y.data
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] > 0:
                    score = float(detections[0, i, j, 0])
                    if ifcut == True:
                        width_plus, hight_plus = img.shape[1] / 4, img.shape[0] / 4
                        pt_cut = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        pt = [int(pt_cut[0] / 2 + width_plus), int(pt_cut[1] / 2 + hight_plus),
                              int(pt_cut[2] / 2 + width_plus), int(pt_cut[3] / 2 + hight_plus)]
                        coords = [score, pt[0], pt[1], pt[2], pt[3]]
                    else:
                        pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                        coords = [score, int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])]
                    coords_list.append(coords)
                    pred_num += 1
                    j += 1
        coords_list = np.array(coords_list)
        coords_list = py_nms(coords_list, thresh, True)
        pred_num = len(coords_list)
        with open(filename, mode='a') as f:
            f.write(str(pred_num))
            for k in range(pred_num):
                #f.write(str(pred_num)+' label: '+label_name+' score: ' +
                #        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                temp = ''
                for c in range(5):
                    if c == 0:
                        temp += str(coords_list[k][c])
                    else:
                        temp += '\t'
                        temp += str(int(coords_list[k][c]))
                # temp = '\t'.join(str(c) for c in coords_list[k])
                f.write('\t' + str(temp) + '\t' + "person")
        with open(filename, mode='a') as f:
            f.write('\n')


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    t_begin = time.time()
    test_voc()
    t_end = time.time()
    print(t_end-t_begin)
