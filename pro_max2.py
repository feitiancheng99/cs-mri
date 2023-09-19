from utils2 import *
import torch
import torch.nn as nn


class UNFOD(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UNFOD, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo
        basicblock = ISTA_2RB_BasicBlock

        for i in range(LayerNo):
            onelayer.append(basicblock())

        self.fcs = nn.ModuleList(onelayer)
        # self.mask = torch.rand(256, 256)
        self.relu = nn.ReLU()

    def forward(self, gt, mask):  # , mask
        xu_real = zero_filled(gt, mask)  # 下采样的实际值
        loss5 = []
        y_gt = tensor2fft(gt)
        x = xu_real
        fk = y_gt
        for i in range(self.LayerNo):
            x = self.fcs[i](x, fk, mask)
            
            fk = res + fk
            
            loss5.append(loss)
            '''
            else:
                xg2 = torch.cat([xg2, x], 1)
            '''
        res = y_gt - tensor2fft(x)
        loss = torch.pow(mask * res, 2).mean().detach()
        x_final = x


        return x_final  ,loss # , xg2


class ISTA_2RB_BasicBlock(torch.nn.Module):
    def __init__(self, ):
        super(ISTA_2RB_BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        kernel_size = 3
        bias = True
        n_feat = 32

        self.conv_D = nn.Conv2d(1, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias)

        modules_body = Residual_Block(n_feat, n_feat, 3, bias=True)  # , res_scale=1
        modules_body2 = Residual_Block(n_feat, n_feat, 3, bias=True)

        self.body = modules_body
        self.body2 = modules_body2

        self.conv_G = nn.Conv2d(n_feat, 1, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv_G2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x, PhiTb, mask):
        lambda_s = self.relu(self.lambda_step)
        
        x_input = x

        x_D = self.conv_D(x_input)  # 特征提取

        x_backward = self.body(x_D)  # 残差块

        x_G2 = self.relu(self.conv_G2(x_backward))

        x_skip = x_backward + x_G2  # 跳跃链接

        x_backward2 = self.body2(x_skip)  # 32

        x_G = self.conv_G(x_backward2)

        x_pred = x_input + x_G
        
        x_pred2 = fft2tensor(tensor2fft(x_pred) - lambda_s * mask*(tensor2fft(x_pred) - PhiTb), mask)

        return x_pred2

        
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):  # , res_scale=1
        super(Residual_Block, self).__init__()
        # self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x
