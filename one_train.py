from utils import *
from torch.nn import init
import scipy.io as sio
from torch.utils.data import Dataset as Data, DataLoader
from natsort import natsorted
import cv2
import glob
import os
import numpy as np
from pro_one import *
from torch.autograd import Variable
from argparse import ArgumentParser

parser = ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default='21700_exp', help='result directory')
parser.add_argument('--n1', type=int, default=217, help='image_shape1')
parser.add_argument('--epoch', type=int, default=20, help='epoch number of training')
parser.add_argument('--layer_num', type=int, default=3, help='phase number')
parser.add_argument('--batch_num', type=int, default=20, help='batch_num')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

epoch_al = args.epoch
n1 = args.n1
n2 = n1

alpha = torch.tensor(math.pow(10, -7) * (n1 * n2)).float().to(device)


class Dataset1(Data):
    def __init__(self, train=False, path=None):
        super(Data, self).__init__()
        self.train = train
        self.ni = natsorted(glob.glob(os.path.join(path, '*')))

    def __getitem__(self, index):
        imgn = cv2.imread(self.ni[index], cv2.IMREAD_GRAYSCALE)
        imgn = np.array(imgn, dtype=np.float32)
        imgn = torch.tensor(imgn)
        imgn = torch.unsqueeze(imgn, dim=0).to(device) / 255.
        #  imgn = np.transpose(imgn, (2, 0, 1))

        return imgn

    def __len__(self):
        return len(self.ni)


a = Dataset1(path='./%s' % args.data_dir)
train_loader2 = DataLoader(dataset=a,
                           batch_size=8,
                           shuffle=True)

model = UNFOD(args.layer_num,n1,n2)
model.to(device)

opt_U = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.MSELoss()

ps_al = []
sp_exp = 0.2
c = Variable(torch.tensor([1])).float().to(device)
sp_now = 1
S_out0 = 0
for step in range(epoch_al):
    ps_all1 = 0
    if sp_now - sp_exp > 0.01:
        c = c * 1.1
    elif sp_now - sp_exp < 0.01:
        c = c * 0.9
    for i, data in enumerate(train_loader2):
        beta = Variable(torch.tensor(5 * step / epoch_al))
        [output,S_out] = model(data,beta,c)  # cnn output
        U_loss = loss_function(output, data)
        opt_U.zero_grad()
        U_loss.backward(retain_graph=True)
        opt_U.step()
        S_out0 = S_out.data
        sp = torch.sum(torch.abs(S_out0)) / (n1 * n2)  # l1正则
        
        sp_now = sp
        ps = psnr1(data, output)
        ps_all1 += ps
        print("epoch：", step, "的第", i, "sparse", sp, "ULOSS", U_loss, "psnr", ps)

    ps_al.append(ps_all1 / args.batch_num)

ps_al = np.array(ps_al)
np.save('./PSNR_%s' % args.data_dir, ps_al)

state_dict = {"model": model.state_dict(), "optimizer": opt_U.state_dict()}
torch.save(state_dict, 'params%s.pth' % args.data_dir)


S = S_out0
S = torch.squeeze(S, 3)
S = torch.squeeze(S, 0)
S_out2 = S.cpu()
S_out2 = S_out2.detach().numpy()
sio.savemat('S_out_%s.mat' % args.data_dir, {'result': S_out2})