"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fourier_2d_fno_shallow_deep import SpectralConv2d
# from fourier_2d_fno_shallow_deep import FNO2d
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer

from torch.optim import Adam

from utilities3 import *
import sys

#seed = int(sys.argv[1])
seed=0
torch.manual_seed(0)
np.random.seed(0)


###############################################################
#fourier layer
###############################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        #x_copy = x.clone()
        size_x, size_y = x.shape[2], x.shape[3]
        #helper_one = torch.ones(x.shape).to(x.device)
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        #helper_ft = torch.fft.rfft2(helper_one)
        #x_copy_ft = x_copy*helper_ft

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        #out_helper_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #out_helper_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(helper_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        #out_helper_ft[:, :, -self.modes1:, :self.modes2]  = self.compl_mul2d(helper_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        #helper = torch.fft.irfft2(out_helper_ft, s=(x.size(-2), x.size(-1)))
        return x #- helper*x_copy

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width,last_width,task_num, nb):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.last_width = last_width
        self.padding = 9
        self.task_num = task_num
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.nb = nb

        self.convlayer = SpectralConv2d(self.width, self.width, self.modes1, self.modes2).cuda()
        #self.w = DenseNet([2,self.width*4,self.width*2,self.width*self.width], torch.nn.ReLU).cuda()
        self.w = nn.Conv1d(self.width, self.width, 1).cuda()

        #self.grid = grid.cuda()
        self.fc1 = nn.ModuleList([nn.Linear(self.width, self.last_width) for i in range(task_num+1)])
        self.fc2 = nn.ModuleList([nn.Linear(self.last_width, 1) for i in range(task_num+1)])



    def forward(self, x,task_idx):
        batchsize = x.shape[0]

        #grid = self.grid.repeat(batchsize,1,1,1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        size_x, size_y = x.shape[2], x.shape[3]
        #R_grid = self.w(grid).view(batchsize,size_x,size_y,self.width,self.width)

        for layer in range(self.nb-1):
            x1 = self.convlayer(x)
            x2 = self.w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x + F.gelu(x1+x2)/self.nb

        x1 = self.convlayer(x)
        x2 = self.w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x +(x1+x2)/self.nb

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1[task_idx](x)
        x = F.gelu(x)
        x = self.fc2[task_idx](x)

        return x


###############################################################
#configs
###############################################################
def read_train_data(input_dir,ntrain,num):
    x_train = []
    y_train = []
    r = 20
    count = 0
    h = int(((421 - 1) / r) + 1)
    s = h
    fcount = 0
    for filename in os.listdir(input_dir):
        print(filename)
        fcount +=1
        if filename.endswith(".mat")  and count<num :
            #    fcount+=1
            try:
                print("current reading")
                print(filename)
                FILE_PATH = os.path.join(input_dir, filename)
                reader = MatReader(FILE_PATH)
                print("reader good")
                x =reader.read_field('coeff')[:ntrain, :, :][:, :s, :s]
                y =reader.read_field('sol')[:ntrain, :, :][:, :s, :s]
                print("x y good")
                if (x.shape[0] ==ntrain and y.shape[0]==ntrain):
                    print("file is good")
                    x_train.append(x)
                    y_train.append(y)
                    print(str(count) +"file finished " + filename)
                    count+=1

            except:
                print("jump this file")
            #    os.remove(FILE_PATH)
            #   print("removed the file")

    if( len(x_train)==num and len(y_train)==num):
        print("Good")
    else:
        return False
    x_train_mixed = torch.cat([item for item in x_train], 0)
    y_train_mixed = torch.cat([item for item in y_train], 0)
    return x_train_mixed, y_train_mixed



# input
TEST_PATH = '../data/Darcy/Meta_data_f_test/output3_12_train_1000_change_f_31200003.mat'
#model name
train_ratio = "f"
test_ratio = "3_12"
cuda_flag = True
#if cuda_flag:
#   torch.cuda.set_device(1)


train_dir = '../data/Darcy/Meta_data_f_500_22_22'
ntrain_pertask =100
task_num =100

model_name = train_ratio+'batch_4_Width64_Q_2layer_train_task1-'+str(task_num)+'_'+str(ntrain_pertask)+'_test_3_800_with_norm_train_model'
if cuda_flag:
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)
RESULT_PATH = '../results/train_' + train_ratio + '_test_' + test_ratio + '/subtask_'+train_ratio+'/' + model_name + '.mat'
#MODEL_PATH = '../models/train_' + train_ratio + '_test_' + test_ratio + '/' + model_name
r = 20
h = int(((421 - 1) / r) + 1)
s = h
dx = 1/(s-1)
ntest = 200
ntest_pretrain =100
x_train, y_train = read_train_data(train_dir, ntrain_pertask,task_num)
l2cons_x_train = dx*torch.norm(x_train,p='fro',dim=(1,2)).reshape(-1,1,1)
x_train  = torch.div(x_train,l2cons_x_train)
y_train  = torch.div(y_train,l2cons_x_train)
ntrain = task_num*ntrain_pertask
# flag variable to indicate whether need to train B first
train_flag = True
batch_size =50
learning_rate = 0.003

epochs = 500
epochs_test =2000
step_size = 100
gamma = 0.5

modes = 12
width = 32
last_width =int(sys.argv[1])
print("ntrain",ntrain,"ntest",ntest,"batch_size",batch_size,"learning_rate",learning_rate,"step_size",step_size,"gamma",gamma,"modes",modes,"width",width)




################################################################
# load data and data normalization
################################################################
reader = MatReader(TEST_PATH)
reader.load_file(TEST_PATH)
print(reader.read_field('coeff').shape)
x_test_pretrain = reader.read_field('coeff')[:ntest_pretrain, :, :][:, :s, :s]
y_test_pretrain = reader.read_field('sol')[:ntest_pretrain, :, :][:, :s, :s]
x_test = reader.read_field('coeff')[300:, :, :][:, :s, :s]
y_test = reader.read_field('sol')[300:, :, :][:, :s, :s]

l2cons_x_test_pretrain = dx*torch.norm(x_test_pretrain,p='fro',dim=(1,2)).reshape(-1,1,1)
x_test_pretrain  = torch.div(x_test_pretrain,l2cons_x_test_pretrain)
y_test_pretrain  = torch.div(y_test_pretrain,l2cons_x_test_pretrain)

l2cons_x_test = dx*torch.norm(x_test,p='fro',dim=(1,2)).reshape(-1,1,1)
x_test  = torch.div(x_test,l2cons_x_test)
y_test  = torch.div(y_test,l2cons_x_test)

print(x_train.shape)
print(x_test.shape)
print(x_test_pretrain.shape)

y_train_normalizers = []
# task wise normalizer
for t in range(task_num):
    x = x_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:]
    y = y_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:]
    x_normalizer = UnitGaussianNormalizer(x)
    x_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:] = x_normalizer.encode(x)
    y_normalizer = UnitGaussianNormalizer(y)
    y_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:] = y_normalizer.encode(y)
    y_train_normalizers.append(y_normalizer)

x_test_normalizer = UnitGaussianNormalizer(x_test_pretrain)
x_test = x_test_normalizer.encode(x_test)
x_test_pretrain = x_test_normalizer.encode(x_test_pretrain)
print(x_test_pretrain.shape)
#
y_test_normalizer = UnitGaussianNormalizer(y_test_pretrain)
y_test_pretrain = y_test_normalizer.encode(y_test_pretrain)
if cuda_flag:
    y_test_normalizer.cuda()


grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)

x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
x_test_pretrain = torch.cat([x_test_pretrain.reshape(ntest_pretrain,s,s,1), grid.repeat(ntest_pretrain,1,1,1)], dim=3)


train_loader = []
for t in range(task_num):
    train_loader.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:], y_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:],l2cons_x_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:]), batch_size=batch_size,
                                                    shuffle=True))

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test,l2cons_x_test), batch_size=batch_size,
                                          shuffle=False)
test_pretrain_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_pretrain, y_test_pretrain,l2cons_x_test_pretrain), batch_size=batch_size,
                                                   shuffle=True)
################################################################
# training and evaluation
################################################################

def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def LR_schedule(learning_rate,steps,scheduler_step,scheduler_gamma):
    #print(steps//scheduler_step)
    return learning_rate*np.power(scheduler_gamma,(steps//scheduler_step))





base_dir = './v3_-3_deep_models_trained_l2norm_taskwise_new/batch_size_%d_res_22_22_%d_%d_2D_Darcy_FNO_%d/shatodeep_samew_ntrain%d_NKN_s%d' %(batch_size,width,last_width,task_num,ntrain_pertask,s)
if not os.path.exists(base_dir):
    os.makedirs(base_dir);

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
total_step = 0
test_l2 = 0.0

for nlayer in range(4):
    nb = 2**nlayer
    print("nlayer: %d" % nb)
    model = FNO2d(modes, modes, width,last_width,task_num,nb)
    if nb != 1:
        restart_nb = nb//2
        model_filename_restart = '%s/FNO_NKN_depth%d.ckpt' % (base_dir,restart_nb)
        model.load_state_dict(torch.load(model_filename_restart))

    print(model)
    print(count_params(model))

    train_flag =True
    if cuda_flag:
        model.cuda()
    optimizer_2 =  Adam([{'params': model.fc2[:99].parameters()},{'params': model.fc1[:99].parameters()}], lr=learning_rate, weight_decay=1e-7)
    optimizer_1 = torch.optim.Adam( [{'params': model.convlayer.parameters()},{'params': model.fc0.parameters()}], lr=learning_rate, weight_decay=1e-5)
    model_filename= '%s/FNO_NKN_depth%d.ckpt' % (base_dir, nb)
    # model_filename += str(9)
    # print("load the model from %s"%model_filename)
    # model.load_state_dict(torch.load(model_filename))
    # train_flag = False
    model_count=0
    if(train_flag == True):
        train_loss_min = 100

        for ep in range(epochs):
            optimizer_1 = scheduler(optimizer_1, LR_schedule(learning_rate, ep, step_size, gamma))
            optimizer_2 = scheduler(optimizer_2, LR_schedule(learning_rate, ep, step_size, gamma))
            print("training model ")
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for i in range(int(ntrain_pertask/batch_size)):
                losses =[]
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                for task_idx in range(task_num):
                    x, y ,l2y_train= next(iter(train_loader[task_idx]))
                    if cuda_flag:
                        x, y ,l2y_train= x.cuda(), y.cuda(),l2y_train.cuda()
                    torch.cuda.empty_cache()
                    out = model(x, task_idx).reshape(batch_size, s, s)
                    y_normalizer = y_train_normalizers[task_idx]
                    y_normalizer.cuda()

                    out = y_normalizer.decode(out)*l2y_train
                    y = y_normalizer.decode(y)*l2y_train
                    losses.append(myloss(out.view(batch_size, -1), y.view(batch_size, -1)))

                torch.cuda.empty_cache()
                (sum(losses)/len(losses)).backward()
                optimizer_2.step()
                optimizer_1.step()
                train_l2 += sum(losses).item()


            model.eval()
            if train_loss_min > train_l2/ntrain:
                train_loss_min = train_l2/ntrain
                print("save a better model now ")
                torch.save(model.state_dict(),model_filename+str(model_count%10))
                torch.save(model.state_dict(),model_filename)
                model_count +=1
                test_l2 = 0.0
                with torch.no_grad():
                    for x, y,l2y_test in test_loader:
                        x, y,l2y_test = x.cuda(), y.cuda(),l2y_test.cuda()

                        out = model(x,task_num).reshape(batch_size, s, s)
                        out = y_test_normalizer.decode(out)*l2y_test
                        y = y*l2y_test

                        test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
                    test_l2 /= ntest

            train_l2 /= ntrain

            t2 = default_timer()
            print('depth: %d epochs: [%d/%d]  running time:%.3f  current training error: %f best training error: %f best test error: %f' % (nb, ep,epochs, t2-t1, train_l2,train_loss_min, test_l2))
            total_step += 1

    optimizer_test =  Adam([{'params': model.fc2[task_num].parameters()},{'params': model.fc1[task_num].parameters()}], lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler_test =  torch.optim.lr_scheduler.StepLR(optimizer_test, step_size=step_size, gamma=gamma)
    test_num = ntest_pretrain
    for ep in range(epochs_test):
    # meta test train last layer
        print("meta test train here ")
        model.train()
        test_pretrain_l2 = 0
        t1 = default_timer()
        for  x,y,l2y_test_pretrain  in test_pretrain_loader:
            optimizer_test.zero_grad()
            if cuda_flag:
                x,y ,l2y_test_pretrain= x.cuda(),y.cuda(),l2y_test_pretrain.cuda()
            out = model(x,task_num).reshape(batch_size, s, s)
            out = y_test_normalizer.decode(out)*l2y_test_pretrain
            y = y_test_normalizer.decode(y)*l2y_test_pretrain
            test_loss = myloss(out.view(batch_size,-1),y.view(batch_size,-1))
            test_pretrain_l2 += test_loss.item()
            test_loss.backward()
            optimizer_test.step()
        scheduler_test.step()


        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y,l2y_test in test_loader:
                x, y,l2y_test = x.cuda(), y.cuda(),l2y_test.cuda()

                out = model(x,task_num).reshape(batch_size, s, s)
                out = y_test_normalizer.decode(out)*l2y_test
                y = y*l2y_test
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()


        test_l2 /= ntest
        test_pretrain_l2/=test_num
        t2 = default_timer()
        print("epoch num ",ep,"time/epoch " ,t2 - t1, "meta test loss ", test_pretrain_l2,"test loss ",test_l2)












