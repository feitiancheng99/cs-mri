from utils2 import *
from torch.nn import init
import scipy.io as sio
from torch.utils.data import Dataset as Data, DataLoader
from natsort import natsorted
import cv2
import glob
import os
import numpy as np
import pro_max2
import gc
import torchvision.transforms as transforms
from argparse import ArgumentParser

parser = ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default='sj2561', help='result directory')
parser.add_argument('--epoch', type=int, default=500, help='epoch number of training')
parser.add_argument('--step_num', type=int, default=5, help='step number')
parser.add_argument('--batch_num', type=int, default=100, help='batch_num')
args = parser.parse_args()

num_all = 50
epoch_al = args.epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

S_INPUT = sio.loadmat('./vd5.mat')  #  './S_out_%s.mat' % args.data_dir
S = S_INPUT['result']  #  result
S = transforms.ToTensor()(S).to(device)
S = torch.unsqueeze(S, 3)*255

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
b = Dataset1(path='./%s_test' % args.data_dir)

train_loader2 = DataLoader(dataset=a,
                           batch_size=8,
                           shuffle=True)

train_loader3 = DataLoader(dataset=b,
                           batch_size=1,
                           shuffle=False)

model = pro_max2.UNFOD(args.step_num)
model.to(device)
'''
checkpoint = torch.load('params%s_1.pth' % args.data_dir)
model.load_state_dict(checkpoint['model'])
'''
opt_U = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt_U, step_size=100, gamma=0.9)
loss_function = nn.MSELoss()
loss_function2 = nn.L1Loss()

ps_al = []
U_al = []
for step in range(epoch_al):
    ps_all1 = 0
    U_all1 = 0
    for i, data in enumerate(train_loader2):
        output = model(data, S)  # cnn output
        U_loss = loss_function(output, data)     

        opt_U.zero_grad()
        U_loss.backward()
        opt_U.step()
        
        ps = psnr1(data, output)
        ps_all1 += ps
        # U_all1 += U
        print("epoch：", step, "的第", i, "LOSS", U_loss,"PSNR", ps)
    ps_al.append(ps_all1 / args.batch_num)
    # U_al.append(U_all1 / args.batch_num)


    scheduler.step()

ps_al = np.array(ps_al)
np.save('./PSNR_1_%s' % args.data_dir, ps_al)
'''
U_al = np.array(U_al)
np.save('./K_1_%s' % args.data_dir, U_al)
'''

'''
x_out = 0.
ps_all2 = 0
for i, data1 in enumerate(train_loader3):
    [out1,k] = model(data1, S)
    
    ps0 = psnr(out1, data1)
    ps_all2 += ps0
    print(ps0)
   
    # x_out2 = torch.cat([out0, out1, data1], 1)
    x_out2 = torch.cat([out1, data1], 1)
    
    if i == 0:
        x_out = x_out2
    else :
        x_out = torch.cat([x_out, x_out2], 1)
    print(x_out.shape)
   


x_out = torch.squeeze(x_out, 0)
x_out = x_out.cpu()
x_out = x_out.detach().numpy()
sio.savemat('x_out_%s.mat' % args.data_dir, {'result': x_out})
'''
# print(ps_all2 / num_all)

state_dict = {"model": model.state_dict(), "optimizer": opt_U.state_dict()}
torch.save(state_dict, 'params%s_1.pth' % args.data_dir)

