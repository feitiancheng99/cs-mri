from utils import *
import torch
import torch.nn as nn

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, gama):  # input=sr1
        ctx.save_for_backward(input, k, gama)
        out1 = input.new(input.size())
        out1[input >= 0] = 1
        out1[input < 0] = 0
        return out1

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output)
        input, k, gama = ctx.saved_tensors
        grad_input = ((5 + k) * (1 - torch.pow(torch.tanh(input * 2 * (5 + k)), 2))) * (grad_output+(1e-7*gama))
        # print(grad_input)
        return grad_input, None, None


class SNet(torch.nn.Module):
    def __init__(self, n1, n2, flag_1D=False):
        super(SNet, self).__init__()
        self.S = nn.Parameter(torch.zeros(1, n1, n2, 1), requires_grad=True) if not flag_1D else \
            nn.Parameter(torch.zeros(1,n1,1), requires_grad=True)
        self.act1 = BinaryQuantize.apply
        self.flag_1D = flag_1D
        self.n2 = n2

    def forward(self, i, ga):
        if self.flag_1D:
            ga = self.n2*ga
            
        S = self.act1(self.S, i, ga)
        
        if self.flag_1D:
            S_1D = S.unsqueeze(2)  # 最后加一维  256,1
            S = torch.cat([S_1D] * self.n2, dim=2)  # 256*256
            # S = S.unsqueeze(0).unsqueeze(-1)
        
        return S

class UNFOD(torch.nn.Module):
    def __init__(self, LayerNo,n1,n2):
        super(UNFOD, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo
        basicblock = ISTA_2RB_BasicBlock
        for i in range(LayerNo):
            onelayer.append(basicblock())

        # self.fcs = nn.ModuleList(onelayer)
        # self.mask = torch.rand(256, 256)
        self.mask=SNet(n1,n2)

    def forward(self, gt,i,ga):  # , mask
        # mask = self.mask
        mask=self.mask(i,ga)
        xu_real = zero_filled(gt, mask)  # 下采样的实际值

        x = xu_real

        for i in range(self.LayerNo):
            x = self.fcs[i](x, xu_real, mask)

        x_final = x
       
        return x_final, mask


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
        x = x - lambda_s * zero_filled(x, mask)
        x = x + lambda_s * PhiTb
        x_input = x

        x_D = self.conv_D(x_input)  # 特征提取

        x_backward = self.body(x_D)  # 残差块

        x_G2 = self.conv_G2(x_backward)

        x_skip = x_backward + x_G2  # 跳跃链接

        x_backward2 = self.body2(x_skip)  # 32

        x_G = self.conv_G(x_backward2)

        x_pred = x_input + x_G

        return x_pred


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
