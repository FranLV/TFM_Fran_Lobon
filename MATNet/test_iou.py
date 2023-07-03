import torch
from torchvision import transforms

import os
import shutil
import glob
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
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

from array import array

def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())

def sorted_by_number(str):
    return int(os.path.basename(str)[:-4])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MatNet Testing')
    parser.add_argument('-i','--input_dir', help='Input directory in the root only images rgb or so', required=True)
    parser.add_argument('-t','--gt_dir', help='Ground Truth directory, each image must have the index in the image name', required=False)
    args = parser.parse_args()

    directories = [f.name for f in os.scandir(args.input_dir) if f.is_dir()]


    for folder in directories:
        imagefiles = sorted(glob.glob(os.path.join(args.input_dir, folder, '*.png')), key=sorted_by_number)
        gtfiles = sorted(glob.glob(os.path.join(args.gt_dir, folder, "gt", '*.png')),key=sorted_by_number)
        
        gt_index = 0  

        save_folder = '{}/{}/iou'.format(args.input_dir, folder)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        

        threshold_values = np.arange(0, 100, 5)

        threshold_values = np.append(threshold_values, [100])

        iou_values = [ [0]*len(gtfiles) for i in range(len(threshold_values))]

        for index, threshold in enumerate(threshold_values): 

            save_folder_threshold = os.path.join(save_folder, "{}".format(threshold))
            os.makedirs(save_folder_threshold)

            gt_index = 0 

            gt_indexes = np.zeros(len(gtfiles))

            with torch.no_grad():
                for imagefile in imagefiles:
                    image = Image.open(imagefile).convert("L")
                    
                    if gt_index < len(gtfiles) and os.path.basename(gtfiles[gt_index])[:-4] == os.path.basename(imagefile)[:-4]:
                        gt = Image.open(gtfiles[gt_index])
                        gt = np.array(gt)                   

                        ret, image_binarized = cv.threshold(np.asarray(image), (threshold/100)*255, 255, cv.THRESH_BINARY)
            
                        # IOU
                        iou_measure = iou.db_eval_iou(gt,image_binarized)
                        print("Ground truth. IOU: ", iou_measure)                   
                        
                        iou_values[index][gt_index] = iou_measure

                        gt_indexes[gt_index] = os.path.basename(gtfiles[gt_index])[:-4]
                        
                        gt_index = gt_index + 1
                        


                    print(os.path.basename(imagefile)[:-4])

                    ret, cv_image_binarized = cv.threshold(np.asarray(image), (threshold/100)*255, 255, cv.THRESH_BINARY)

                    image_binarized = Image.fromarray(cv_image_binarized).convert('L')

                    save_file = os.path.join(save_folder_threshold,
                                            os.path.basename(imagefile)[:-4] + '.png')
                    
                    image_binarized.save(save_file)
        


        iou_values_list = np.array(iou_values)  # Convertir la lista en un arreglo NumPy
        iou_values_list_no_0 = iou_values_list[iou_values_list != 0.0]  # Filtrar los valores diferentes de 0
        iou_values_no_0 = np.split(iou_values_list_no_0, np.cumsum(np.sum(iou_values_list != 0, axis=1)))[:-1]
        
        iou_median = np.zeros(len(iou_values_no_0))
        for i in range(len(iou_values_no_0)):
            if len(iou_values_no_0[i]) == 0:
                iou_values_no_0[i] = [0.0]

            iou_median[i] = np.median(iou_values_no_0[i], axis=0) * 100

        # iou_median = np.median(iou_values, axis=1) * 100


        save_iou = os.path.join(save_folder, "IOU_{}.csv".format(folder))                    
        iou_file = open(save_iou, "a")
        iou_file.write("Umbral\\Fograma,{},Media,Total Fotogramas\n".format(",".join(str(x) for x in gt_indexes)))

        for index in range(len(iou_values)):
            iou_file.write("{},{},{},{}\n".format(index*5, ",".join(str(x) for x in iou_values[index]),iou_median[index],len(iou_values_no_0[index])))
        
        iou_file.close()


        colours = ["#eeffba","#5d8700"]
        average = 60

        # Colormap - Build the colour maps
        cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
        norm = mpl.colors.Normalize(iou_median.min(), iou_median.max()) # linearly normalizes data into the [0.0, 1.0] interval

        fig = plt.figure()

        container = plt.bar(threshold_values, iou_median, color=cmap(norm(iou_median)), width=2)
        plt.bar_label(container,labels=[f'{e:,d}' for e in iou_median.astype(int)], padding=3, color='black', fontsize=8) 
        # plt.axhline(y=average, color = 'grey', linewidth=3)
        plt.xticks(threshold_values)
        font = {    
            'family': 'Computer Modern',
            'color':  'black',
            'weight': 'normal',
            'size': 9,
            }
        title_font = {    
            'family': 'Computer Modern',
            'color':  'black',
            'weight': 'normal',
            'size': 10,
            }
        plt.ylabel('IOU (%)', font)
        plt.xlabel('Umbral (%)', font)
        plt.title(folder, title_font)
        # plt.show()

        # plt.gca().set_position([0, 0, 1, 1])

        fig.savefig(os.path.join(save_folder, folder + '_grafica.svg'), bbox_inches='tight')
