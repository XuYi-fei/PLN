import torch
from torch import  nn
from torchvision import models
from torchvision import transforms
from .net_old import Conv_1, Conv_2,Dialation

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv_1 = nn.Conv2d(2048,3328,kernel_size=1)
        # The old Conv_1()
        self.conv_2 = nn.Conv2d(3328, 1536, kernel_size=1)
        self.conv_3 = nn.Conv2d(1536, 1536, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        # old Conv_1 done

        # The old Conv_2()
        self.conv_5a = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        self.conv_5b = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        self.conv_5c = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        self.conv_5d = nn.Conv2d(1536, 1536, kernel_size=3,padding=1)
        self.conv_6a = nn.Conv2d(1536, 204, kernel_size=1)
        self.conv_6b = nn.Conv2d(1536, 204, kernel_size=1)
        self.conv_6c = nn.Conv2d(1536, 204, kernel_size=1)
        self.conv_6d = nn.Conv2d(1536, 204, kernel_size=1)
        # old Conv_2 done

        # The old dialation
        self.dialation_a1 = nn.Conv2d(204, 204, kernel_size=3, dilation=2, padding=2)
        self.dialation_a2 = nn.Conv2d(204, 204, kernel_size=3, dilation=2, padding=2)
        self.dialation_a3 = nn.Conv2d(204, 204, kernel_size=3, dilation=2, padding=2)
        self.dialation_a4 = nn.Conv2d(204, 204, kernel_size=3, dilation=2, padding=2)
        self.dialation_b1 = nn.Conv2d(204, 204, kernel_size=3, dilation=4, padding=4)
        self.dialation_b2 = nn.Conv2d(204, 204, kernel_size=3, dilation=4, padding=4)
        self.dialation_b3 = nn.Conv2d(204, 204, kernel_size=3, dilation=4, padding=4)
        self.dialation_b4 = nn.Conv2d(204, 204, kernel_size=3, dilation=4, padding=4)
        self.dialation_c1 = nn.Conv2d(204, 204, kernel_size=3, dilation=8, padding=8)
        self.dialation_c2 = nn.Conv2d(204, 204, kernel_size=3, dilation=8, padding=8)
        self.dialation_c3 = nn.Conv2d(204, 204, kernel_size=3, dilation=8, padding=8)
        self.dialation_c4 = nn.Conv2d(204, 204, kernel_size=3, dilation=8, padding=8)
        self.dialation_d1 = nn.Conv2d(204, 204, kernel_size=3, dilation=16, padding=16)
        self.dialation_d2 = nn.Conv2d(204, 204, kernel_size=3, dilation=16, padding=16)
        self.dialation_d3 = nn.Conv2d(204, 204, kernel_size=3, dilation=16, padding=16)
        self.dialation_d4 = nn.Conv2d(204, 204, kernel_size=3, dilation=16, padding=16)
        self.dialation_e1 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_e2 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_e3 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_e4 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_f1 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_f2 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_f3 = nn.Conv2d(204, 204, kernel_size=1)
        self.dialation_f4 = nn.Conv2d(204, 204, kernel_size=1)
        # old dialation done
        self.Init_tensors()
        self.inception = models.inception_v3(pretrained=True)
        self.sequence_commonn = nn.Sequential(self.conv_1,self.conv_2,self.conv_3,self.conv_4)
        self.sequence_1 = nn.Sequential(self.conv_5a,self.conv_6a,self.dialation_a1,self.dialation_b1,
                                      self.dialation_c1,self.dialation_d1,self.dialation_e1,self.dialation_f1)
        self.sequence_2 = nn.Sequential(self.conv_5b, self.conv_6b, self.dialation_a2, self.dialation_b2,
                                        self.dialation_c2, self.dialation_d2, self.dialation_e2, self.dialation_f2)
        self.sequence_3 = nn.Sequential(self.conv_5c, self.conv_6c, self.dialation_a3, self.dialation_b3,
                                        self.dialation_c3, self.dialation_d3, self.dialation_e3, self.dialation_f3)
        self.sequence_4 = nn.Sequential(self.conv_5d, self.conv_6d, self.dialation_a4, self.dialation_b4,
                                        self.dialation_c4, self.dialation_d4, self.dialation_e4, self.dialation_f4)
    def Init_tensors(self):
        torch.nn.init.xavier_uniform_(self.conv_1.weight)
        torch.nn.init.xavier_uniform_(self.conv_2.weight)
        torch.nn.init.xavier_uniform_(self.conv_3.weight)
        torch.nn.init.xavier_uniform_(self.conv_4.weight)
        torch.nn.init.xavier_uniform_(self.conv_5a.weight)
        torch.nn.init.xavier_uniform_(self.conv_5b.weight)
        torch.nn.init.xavier_uniform_(self.conv_5c.weight)
        torch.nn.init.xavier_uniform_(self.conv_5d.weight)
        torch.nn.init.xavier_uniform_(self.conv_6a.weight)
        torch.nn.init.xavier_uniform_(self.conv_6b.weight)
        torch.nn.init.xavier_uniform_(self.conv_6c.weight)
        torch.nn.init.xavier_uniform_(self.conv_6d.weight)
        torch.nn.init.xavier_uniform_(self.dialation_a1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_a2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_a3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_a4.weight)
        torch.nn.init.xavier_uniform_(self.dialation_b1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_b2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_b3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_b4.weight)
        torch.nn.init.xavier_uniform_(self.dialation_c1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_c2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_c3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_c4.weight)
        torch.nn.init.xavier_uniform_(self.dialation_d1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_d2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_d3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_d4.weight)
        torch.nn.init.xavier_uniform_(self.dialation_e1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_e2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_e3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_e4.weight)
        torch.nn.init.xavier_uniform_(self.dialation_f1.weight)
        torch.nn.init.xavier_uniform_(self.dialation_f2.weight)
        torch.nn.init.xavier_uniform_(self.dialation_f3.weight)
        torch.nn.init.xavier_uniform_(self.dialation_f4.weight)
        return


    def forward(self,x):
        # x should be (N,C,H,W)
        # here is (N,C,512,512) and x is a tensor
        N = x.shape[0]
        x.requires_grad = True
        # C = x.shape[1]
        # batch = nn.BatchNorm2d(C)
        # x = batch(x)
        x = self.inception(x)[0]
        x = self.sequence_commonn(x)
        input1 = x.clone()
        input2 = x.clone()
        input3 = x.clone()
        input4 = x.clone()
        output1 = self.sequence_1(input1)
        output2 = self.sequence_2(input2)
        output3 = self.sequence_3(input3)
        output4 = self.sequence_4(input4)
        # output1 = torch.sigmoid(output1)
        # output2 = torch.sigmoid(output2)
        # output3 = torch.sigmoid(output3)
        # output4 = torch.sigmoid(output4)
        output1 = torch.abs(output1)
        output2 = torch.abs(output2)
        output3 = torch.abs(output3)
        output4 = torch.abs(output4)
        # output1 = torch.where(output1 > 1, torch.full_like(output1,1.), output1)
        # output2 = torch.where(output1 > 1, torch.full_like(output2,1.), output2)
        # output3 = torch.where(output1 > 1, torch.full_like(output3,1.), output3)
        # output4 = torch.where(output1 > 1, torch.full_like(output4,1.), output4)

        # leakyrelu = torch.nn.LeakyReLU(0.01)
        # output1 = leakyrelu(output1)
        # output2 = leakyrelu(output2)
        # output3 = leakyrelu(output3)
        # output4 = leakyrelu(output4)
        # output1 = torch.where(output1 < 0, torch.full_like(output1, 0.), output1)
        # output2 = torch.where(output1 < 0, torch.full_like(output2, 0.), output2)
        # output3 = torch.where(output1 < 0, torch.full_like(output2, 0.), output3)
        # output4 = torch.where(output1 < 0, torch.full_like(output2, 0.), output4)
        # selu = nn.SELU()
        # output1 = selu(output1)
        # output2 = selu(output2)
        # output3 = selu(output3)
        # output4 = selu(output4)

        # softmax = nn.Softmax(dim=2)
        # output1 = output1.view((N,204,14*14))
        # output1 = softmax(output1)
        # output2 = output2.view((N,204,14*14))
        # output2 = softmax(output2)
        # output3 = output3.view((N,204,14*14))
        # output3 = softmax(output3)
        # output4 = output4.view((N,204,14*14))
        # output4 = softmax(output4)
        # output1 = output1.view((N,204,14,14))
        # output2 = output2.view((N,204,14,14))
        # output3 = output3.view((N,204,14,14))
        # output4 = output4.view((N,204,14,14))

        # relu = nn.ReLU6()
        # output1 = relu(output1) / 6
        # output2 = relu(output2) / 6
        # output3 = relu(output3) / 6
        # output4 = relu(output4) / 6
        # here output should be (N,204,14,14)

        return output1,output2,output3,output4



