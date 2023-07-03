import torch
from torchvision import transforms

import os
import glob
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import cv2 as cv

# from scipy.misc import imresize

# import cv2

from modules.MATNet import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch
import measures.jaccard as iou

import numpy as np

import time

import argparse

def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())

def sorted_by_number(str):
    return int(os.path.basename(str)[:-4])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MatNet Testing')
    parser.add_argument('-i','--input_dir', help='Input directory in the root only images rgb or so, and root/flow/ to the flow images', required=True)
    parser.add_argument('-o','--output_dir', help='Output directory', required=True)
    parser.add_argument('--use_gpu', help='Use GPU or CPU', required=False, default=False, action="store_true")
    args = parser.parse_args()

    inputRes = (473, 473)
    use_flip = False
    use_cuda = args.use_gpu

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])

    model_name = 'matnet' # specify the model name
    epoch = 0 # specify the epoch number
    davis_result_dir = args.output_dir

    encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args =\
        load_checkpoint_epoch(model_name, epoch, False, False)
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    encoder.train(False)
    decoder.train(False)

    directories = [f.name for f in os.scandir(args.input_dir) if f.is_dir()]

    for folder in directories:
        flow_dir = os.path.join(folder, "flow")

        imagefiles = sorted(glob.glob(os.path.join(args.input_dir, folder, '*.png')), key=sorted_by_number)
        flowfiles = sorted(glob.glob(os.path.join(args.input_dir, flow_dir, '*.png')), key=sorted_by_number)

        save_folder = '{}/{}'.format(davis_result_dir, folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_time = os.path.join(save_folder, "inference_time.text")
        time_file = open(save_time, "a")

        with torch.no_grad():
            for imagefile, flowfile in zip(imagefiles, flowfiles):
                image = Image.open(imagefile).convert('RGB')
                flow = Image.open(flowfile).convert('RGB')

                width, height = image.size

                image = image.resize(inputRes)
                flow = flow.resize(inputRes)

                image = image_transforms(image)
                flow = image_transforms(flow)

                image = image.unsqueeze(0)
                flow = flow.unsqueeze(0)

                if use_cuda:
                    image, flow = image.cuda(), flow.cuda()



                t_total = time.perf_counter()


                r5, r4, r3, r2 = encoder(image, flow)
                mask_pred, bdry_pred, p2, p3, p4, p5 = decoder(r5, r4, r3, r2)


                t_total = time.perf_counter() - t_total

                time_file.write("{}\n".format(t_total))



                if use_flip:
                    image_flip = flip(image, 3)
                    flow_flip = flip(flow, 3)
                    r5, r4, r3, r2 = encoder(image_flip, flow_flip)
                    mask_pred_flip, bdry_pred_flip, p2, p3, p4, p5 =\
                        decoder(r5, r4, r3, r2)

                    mask_pred_flip = flip(mask_pred_flip, 3)
                    bdry_pred_flip = flip(bdry_pred_flip, 3)

                    mask_pred = (mask_pred + mask_pred_flip) / 2.0
                    bdry_pred = (bdry_pred + bdry_pred_flip) / 2.0

                mask_pred = mask_pred[0, 0, :, :]


                print(os.path.basename(imagefile)[:-4])

                mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')


                save_file = os.path.join(save_folder,
                                        os.path.basename(imagefile)[:-4] + '.png')
                mask_pred = mask_pred.resize((width, height))
                mask_pred.save(save_file)

        # iou_file.close()
        time_file.close()


