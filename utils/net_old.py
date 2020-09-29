from torch import nn
from torch.autograd import Variable
import torch
from torchvision import models
from tqdm import tqdm

class Conv_1(nn.Module):
    def __init__(self):
        super(Conv_1,self).__init__()
        self.conv_a = nn.Conv2d(3328, 1536, kernel_size=1)
        self.conv_b = nn.Conv2d(1536, 1536, kernel_size=3, padding=1)
        self.conv_c = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
    def __call__(self, x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.conv_c(x)
        return x

class Conv_2(nn.Module):
    def __init__(self):
        super(Conv_2,self).__init__()
        self.conv_a = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        self.conv_b = nn.Conv2d(1536, 204, kernel_size=1)
    def forward(self,x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        return x

class Dialation(nn.Module):
    def __init__(self):
        super(Dialation,self).__init__()
        self.dialation_a = nn.Conv2d(204,204,kernel_size=3,dilation=2,padding=2)
        self.dialation_b = nn.Conv2d(204,204,kernel_size=3,dilation=4,padding=4)
        self.dialation_c = nn.Conv2d(204,204,kernel_size=3,dilation=8,padding=8)
        self.dialation_d = nn.Conv2d(204,204,kernel_size=3,dilation=16,padding=16)
        self.dialation_e = nn.Conv2d(204,204,kernel_size=1)
        self.dialation_f = nn.Conv2d(204,204,kernel_size=1)
        self.sequence = nn.Sequential(self.dialation_a,self.dialation_b,self.dialation_c,self.dialation_d,
                                      self.dialation_e,self.dialation_f)

    def __call__(self, x):
        x = self.sequence(x)
        return x

class PLN(nn.Module):
    # This class is for some basic ways to pass the variable
    def __init__(self):
        super(PLN,self).__init__()
        self.conv_1 = nn.Conv2d(2048,3328,kernel_size=1)
        self.conv_2 = Conv_1()
        self.conv_3a = Conv_2()
        self.conv_3b = Conv_2()
        self.conv_3c = Conv_2()
        self.conv_3d = Conv_2()
        self.conv_4a, self.conv_4b, self.conv_4c, self.conv_4d = Dialation(), Dialation(), Dialation(), Dialation()
    def forward(self,x):
        x = Variable(x)
        x.requires_grad = True
        if len(list(x.size())) == 3:
            x = x.unsqueeze(0)

        # x should be (N,C,H,W)
        inception_v3 = models.inception_v3(pretrained=True)
        x = inception_v3(x)[0]
        # for discription
        x = self.conv_1(x)
        # for discription
        x = self.conv_2(x)
        # for discription
        x1 = self.conv_3a(x)
        x2 = self.conv_3b(x)
        x3 = self.conv_3c(x)
        x4 = self.conv_3d(x)
        #for discription
        x1 = self.conv_4a(x1)
        x2 = self.conv_4b(x2)
        x3 = self.conv_4c(x3)
        x4 = self.conv_4d(x4)
        # Then put all of them to [0,1]
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        x3 = torch.sigmoid(x3)
        x4 = torch.sigmoid(x4)

        return (x1,x2,x3,x4)

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

                if corner_x > center_x and corner_y > center_y:
                    # 左上和中间点
                    x1, y1 = corner_x, corner_y
                    x2, y2 = 2 * center_x - corner_x, 2 * center_y - corner_y
                elif corner_x > center_x and corner_y < center_y:
                    # 左下和中间点
                    x1, y1 = corner_x, 2 * center_y - corner_y
                    x2, y2 = 2 * center_x - corner_x, corner_y
                elif corner_x < center_x and corner_y > center_y:
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