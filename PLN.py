import sys
# sys.path.append('PLN/utils')

import torch
from torchvision import models
from torch.autograd import Variable
import numpy as np

import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
import os
import cv2
from tqdm import tqdm
from torch import nn
from PIL import Image
from utils.visualization import Visualizer
from data.dataset import Dataset
from utils.net_old import PLN, Conv_1, Conv_2, Dialation
from utils.net import Net
from utils.loss import L2_loss_func


def preprocess(img_path):
    # img_list = os.listdir(img_path)
    # all_imgs = []
    # for img in img_list:
    #     img = os.path.join(img_path,img)
    #     im = Image.open(img)
    #     all_imgs.append(im)
    # im = Image.open(img_path)
    trans = transforms.Compose([transforms.ToTensor()])
    im = trans(img_path)
    im.cuda()
    return im

class PLN(nn.Module):
    # This class is for some basic ways to pass the variable
    def __init__(self):
        super(PLN,self).__init__()


    def inference_analyze(self,inference):
        # from inference get the possibilities results, each of the results is [N,20,196,196]
        # The elements of the results refer to the possibilities that two points are connected together in one class
        # inference(N,204,14,14)
        N = inference.shape[0]
        inference = inference.view(inference.shape[0],inference.shape[1],
                                   inference.shape[2]*inference.shape[3])
        corner_points1 = inference[:,:inference.shape[1]//4,:]
        corner_points2 = inference[:,inference.shape[1]//4:inference.shape[1]//4*2,:]
        center_points1 = inference[:,inference.shape[1]//4*2:inference.shape[1]//4*3,:]
        center_points2 = inference[:,inference.shape[1]//4*3:,:]


        possibility_array1 = self.get_poss(corner_points1,center_points1).view(N,20,14,14,14,14)
        possibility_array2 = self.get_poss(corner_points2,center_points1).view(N,20,14,14,14,14)
        possibility_array3 = self.get_poss(corner_points1,center_points2).view(N,20,14,14,14,14)
        possibility_array4 = self.get_poss(corner_points2,center_points2).view(N,20,14,14,14,14)

        return possibility_array1, possibility_array2, possibility_array3, possibility_array4

    def get_poss(self, corner_points, center_points):
        N = corner_points.shape[0]
        t = torch.zeros((N,20,38416))
        t.cuda()
        print('Calculating possibilities for one point...')
        for i in tqdm(range(196)):
            for j in range(196):
                ix , iy = i // 14, i % 14
                sx , sy = j // 14, j % 14
                t[:,:,i * 196 + j] = self.calculate_poss(corner_points[:,:,i],center_points[:,:,j],ix,iy,sx,sy)
        print('Possibilities for one points calculated done!')
        return t

    def calculate_poss(self,corner_point,center_point,ix,iy,sx,sy):
        # corner_point and center point are both (N,51)
        # calculate the single possibility
        poss = corner_point[:,0].unsqueeze(-1) * center_point[:,0].unsqueeze(-1) * corner_point[:,1:21] * center_point[:,1:21]
        L_ij = corner_point[:,23+sx].unsqueeze(-1) * corner_point[:,37+sy].unsqueeze(-1)
        L_st = center_point[:,23+ix].unsqueeze(-1) * center_point[:,37+iy].unsqueeze(-1)
        L = 0.5 * (L_ij * L_st)
        poss *= L
        return poss

    def find_max_index(self, result, top_k):
        # find the topk score in the image

        # result[N,20,196,196]
        result = result.view(result.shape[0], 20*196*196)
        result, result_indices = torch.sort(result, dim=1,descending=True)
        # result = result[:,:,:int(top_k)]
        N = result_indices // (196*196)
        X, Y = (result_indices - N *(196*196)) // 196, (result_indices - N *(196*196)) % 196
        iX, iY, sX, sY = X//14, X % 14, Y // 14, Y % 14
        index_list = []
        for i in range(result.shape[0]):
            list_imgs = []
            for j in range(top_k):
                ix,iy,sx,sy,n, p = iX[i,j], iY[i,j],sX[i,j],sY[i,j],N[i,j], result[i][j]
                # print(iX[i][j])
                # print(ix)
                list_imgs.append((n,ix,iy,sx,sy,p))
            index_list.append(list_imgs)


        return index_list

    def get_bbox(self,pair, res_index):
        # To calculate the bbox for N images
        # pair has two tensors, each one with (N,51,14
        # ,14), pair[0] is corner point, pair[1] is center point
        # res_index[[(),(),(),(),(), ()]....]
        # res_index[N,topk],every element is a turple with 6 elements stand for n,ix,iy,sx,sy,p
        N, top_k = len(res_index), len(res_index[0])
        bboxes = []
        # bboxes is a list, containing N lists, each of them contains top_k turples, which
        # are like (class,x1,y1,x2,y2,confidence)
        # The width and the height of the image is 512 * 512

        w, h = 512, 512
        w_unit, h_unit = 512 / 14, 512 / 14     # w_unit, h_unit stands for w, h for the single grid

        for i in range(N):
            bbox = []
            for j in range(top_k):
                classes, ix, iy, sx, sy, confidence = res_index[i][j]
                corner_x, corner_y = ix * w_unit + pair[0][i][21][ix][iy] * w_unit, iy * h_unit + pair[0][i][22][ix][iy] * h_unit
                center_x, center_y = sx * w_unit + pair[1][i][21][sx][sy] * w_unit, sy * h_unit + pair[1][i][22][sx][sy] * h_unit
                if corner_x == center_x or corner_y == center_y:
                    # 对角线在一条水平或者竖直线上了
                    continue

                if corner_x < center_x and corner_y < center_y:
                    # 左上和中间点
                    x1, y1 = corner_x, corner_y
                    x2, y2 = 2 * center_x - corner_x, 2 * center_y - corner_y
                elif corner_x < center_x and corner_y > center_y:
                    # 左下和中间点
                    x1, y1 = corner_x, 2 * center_y - corner_y
                    x2, y2 = 2 * center_x - corner_x, corner_y
                elif corner_x > center_x and corner_y < center_y:
                    # 右上和中心点
                    x1, y1 = 2 * center_x - corner_x, corner_y
                    x2, y2 = corner_x, 2 * center_y - corner_y
                else:
                    # 右下和中心点
                    x1, y1 = 2 * center_x - corner_x,  2 * center_y - corner_y
                    x2, y2 = corner_x, corner_y

                bbox.append((classes,int(x1),int(y1),int(x2),int(y2),confidence))
            bboxes.append(bbox)

        return bboxes


class PLNet(PLN):
    def __init__(self,tf = transforms.Resize((512,512))):
        super(PLNet,self).__init__()
        self.tf = tf
        self.epoch = 400
        self.grid_w, self.grid_h = 512 /14, 512 / 14
        self.index2class = {}
        self.class2index = {"aeroplane":0,"bicycle":1,"bird":2,"boat":3,"bottle":4,"bus":5,"car":6,"cat":7,
                            "chair":8,"cow":9,"diningtable":10,"dog":11,"horse":12,"motorbike":13,"person":14,
                            "pottedplant":15,"sheep":16,"sofa":17,"train":18,"tvmonitor":19}
        for k,v in self.class2index.items():
            self.index2class[v] = k


    def main_process(self,img_path, mode = 'test'):
        # mode 'test' or 'train'
        # img: one single img or a img_path to a folder

        if mode == 'test':
            if not os.path.isdir(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.tf(img)
                tensor = preprocess(img)

                # pln = PLN()
                # inference_1, inference_2, inference_3, inference_4 = pln.forward(tensor)
                #
                # res1a, res1b, res1c, res1d = pln.inference_analyze(inference_1)
                # N = res1a.shape[0]
                #
                # res1a_index, res1b_index, res1c_index, res1d_index = \
                #     pln.find_max_index(res1a, 5), pln.find_max_index(res1b, 5), pln.find_max_index(res1c,5), \
                #     pln.find_max_index(res1d, 5)
                #
                # # pair----> A turple, with each(N,51,14,14)containing the parameters
                # # pair_1a corner_point1 with center_point_1----->res1
                # # I copied the pairs from inference just to get the x and y according to the probabilities that calculated in
                # # the res_indexes, other parameters are useless when doing inference.+
                # pair_1a = (inference_1[:, :inference_1.shape[1] // 4, :, :],
                #            inference_1[:, inference_1.shape[1] // 4 * 2: inference_1.shape[1] // 4 * 3, :, :])
                # # pair_1b corner_point2 with center point 1----->res2
                # pair_1b = (inference_1[:, inference_1.shape[1] // 4:inference_1.shape[1] // 4 * 2, :, :],
                #            inference_1[:, inference_1.shape[1] // 4 * 2: inference_1.shape[1] // 4 * 3, :, :])
                # # pair_1c corner_point1 with center point 2----->res3
                # pair_1c = (inference_1[:, :inference_1.shape[1] // 4, :, :],
                #            inference_1[:, inference_1.shape[1] // 4 * 3:, :, :])
                # # pair_1d corner_point2 with center point 2----->res4
                # pair_1d = (inference_1[:, inference_1.shape[1] // 4:inference_1.shape[1] // 4 * 2, :, :],
                #            inference_1[:, inference_1.shape[1] // 4 * 3:, :, :])
                #
                # visualizer = Visualizer()
                # bboxes = pln.get_bbox(pair_1a, res1a_index)
                # visualizer.draw_box(bboxes[0], img)
            else:
                # if img is a path to dataset
                data= Dataset(img_path,'2007','train',transform=self.tf)
                # To store all the images in the file
                target = []
                tensor_list = []
                imgs = []
                print('Reading imgs from VOC data_train....')
                for idx, i in tqdm(enumerate(data)):
                    if idx > 0:
                        break
                    imgs.append(i[0])
                    tensor_list.append(preprocess(i[0]))
                    target.append(i[1])

                print('Data read over.')

                tensor = torch.stack(tensor_list)


            net = Net()
            net.load_state_dict(torch.load('trained_model.pth'))
            pln = PLN()
            inference_1, inference_2, inference_3, inference_4 = net(tensor)
            # inference[N,51,14,14]
            res1a, res1b, res1c, res1d = pln.inference_analyze(inference_1)
            # res1a[N,20,14,14,14,14]
            N = res1a.shape[0]

            res1a_index, res1b_index, res1c_index, res1d_index = \
                pln.find_max_index(res1a, 1), pln.find_max_index(res1b, 2), pln.find_max_index(res1c, 2), \
                pln.find_max_index(res1d, 2)

            # pair----> A turple, with each(N,51,14,14)containing the parameters
            # pair_1a corner_point1 with center_point_1----->res1
            # I copied the pairs from inference just to get the x and y according to the probabilities that calculated in
            # the res_indexes, other parameters are useless when doing inference.+
            pair_1a = (inference_1[:, :inference_1.shape[1] // 4, :, :],
                       inference_1[:, inference_1.shape[1] // 4 * 2: inference_1.shape[1] // 4 * 3, :, :])
            # pair_1b corner_point2 with center point 1----->res2
            pair_1b = (inference_1[:, inference_1.shape[1] // 4:inference_1.shape[1] // 4 * 2, :, :],
                       inference_1[:, inference_1.shape[1] // 4 * 2: inference_1.shape[1] // 4 * 3, :, :])
            # pair_1c corner_point1 with center point 2----->res3
            pair_1c = (inference_1[:, :inference_1.shape[1] // 4, :, :],
                       inference_1[:, inference_1.shape[1] // 4 * 3:, :, :])
            # pair_1d corner_point2 with center point 2----->res4
            pair_1d = (inference_1[:, inference_1.shape[1] // 4:inference_1.shape[1] // 4 * 2, :, :],
                       inference_1[:, inference_1.shape[1] // 4 * 3:, :, :])

            visualizer = Visualizer()
            bboxes = pln.get_bbox(pair_1a, res1a_index)
            visualizer.draw_box(bboxes[0], imgs[0])

        elif mode == 'train':
            assert os.path.isdir(img_path)
            # if img is single image, it can't be used to train
            data = Dataset(img_path, '2007', 'train', transform=self.tf)
            # To store all the images in the file
            net = Net()
            Loss = L2_loss_func()
            optimzer = optim.Adam(net.parameters(),lr=5*1e-6)
            print('Training.....')
            for epoch in tqdm(range(self.epoch)):
                optimzer.zero_grad()
                tensor_list = []
                target = []
                print('\nEpoch:', epoch)
                print()
                # print('Reading VOC data.....')
                for idx, i in tqdm(enumerate(data)):
                    if idx > 0:
                        break
                    # i[0].save("1.jpg")
                    tensor_list.append(preprocess(i[0]))
                    target.append(i[1])
                target = self.analyze_target(target)
                # print('Data read over.\n')

                tensor = torch.stack(tensor_list)
                # tensor is tensors of imgs


                # inference_1, inference_2, inference_3, inference_4 = pln.forward(tensor)

                inference1, inference2, inference3, inference4 = net(tensor)
                # inference[N,51,14,14]

                N = inference1.shape[0]
                print('max parameter:', max(inference1), "min parameter:",min(inference1))
                loss= Loss(inference1,target) + Loss(inference2,target) + Loss(inference3,target) + Loss(inference4,target)
                print( "total loss:", loss)
                loss.backward()
                optimzer.step()

            torch.save(net.state_dict(), 'trained_model.pth')

            print()


    def analyze_target(self, targets):
        # target has N dicts, each dict has filename, size,  bboxes;
        # bboxes has 6 elements, the former 5 is (x,y), four is corner points and the fifth one is center point,
        # the last one is class_name
        N = len(targets)
        tensor = torch.zeros((N,204,14,14))
        for i in range(N):
            tensor[i,:,:,:] = self.decode_target(targets[i])

        return tensor

    def decode_target(self, target):
        # target is a dict just is an element of targets in function 'analyze_target'
        tensors = torch.zeros((204,14,14))
        for bbox in target['bboxes']:
            class_index = bbox[-1]
            ix1, iy1, x1, y1 = self.decode_xy(bbox[0])
            ix2, iy2, x2, y2 = self.decode_xy(bbox[1])
            ix3, iy3, x3, y3 = self.decode_xy(bbox[2])
            ix4, iy4, x4, y4 = self.decode_xy(bbox[3])
            sx, sy , x, y = self.decode_xy(bbox[4])
            # deal with P
            tensors[0, ix1, iy1] = 1
            tensors[51, ix4, iy4] = 1
            tensors[102, sx, sy] = 1
            tensors[153, sx, sy] = 1
            # deal with classes
            tensors[class_index+1, ix1, iy1] = 1
            tensors[class_index+1+51, ix4, iy4] = 1
            tensors[class_index+1+51*2, sx, sy] = 1
            tensors[class_index+1+51*3, sx, sy] = 1
            # deal with x and y
            tensors[21,ix1,iy1] = x1
            tensors[22,ix1,iy1] = y1
            tensors[21+51, ix4, iy4] = x4
            tensors[22+51, ix4, iy4] = y4
            tensors[21+51*2, sx, sy] = x
            tensors[22+51*2, sx, sy] = y
            tensors[21+51*3, sx, sy] = x
            tensors[22+51*3, sx, sy] = y
            #deal with links
            tensors[23+sx, ix1, iy1] = 1
            tensors[37+sy, ix1, iy1] = 1
            tensors[23+sx+51, ix4, iy4] = 1
            tensors[37+sy+51, ix4, iy4] = 1
            tensors[23+ix1+51*2, sx, sy] = 1
            tensors[37+iy1+51*2, sx, sy] = 1
            tensors[37+ix4+51*3, sx, sy] = 1
            tensors[37+iy4+51*3, sx, sy] = 1
            # done!
        return tensors
    def decode_xy(self,bbox):
        # bbox is (x,y), return ix,iy, x, y
        ix, iy = bbox[0] // self.grid_w, bbox[1] // self.grid_h
        x, y = bbox[0] - self.grid_w * ix, bbox[1] - self.grid_h * iy
        ix, iy = int(ix), int(iy)
        if ix == 14:
            ix = 13
            x = 1.0
        if iy == 14:
            iy = 13
            x = 1.0

        return (int(ix),int(iy),x,y)


if __name__ == "__main__":
    # tf = transforms.Resize((512,512))
    # train_data = datasets.VOCDetection('./VOCdevkit','2007',image_set='train',transform=None, download=True)
    # for i in train_data:
    #     print()
    #     print()
    # d = train_data[0]
    PLNet = PLNet()
    PLNet.main_process('./VOCdevkit',mode='test')



    print()





