import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from typing import  Optional


#Time_seg+LDLinear
class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.LD = LD(kernel_size=kernel_size)
        # self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        self.Linear_Seasonal = Time_seg(self.seq_len, self.pred_len,3,configs.gpu)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        trend_init = self.LD(x)
        seasonal_init = x - trend_init
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
    



class Time_seg(nn.Module):
    def __init__(self,input_len,pred_len,seg_num,device):
        super().__init__()
        self.input_len=input_len
        self.pred_len=pred_len
        self.seg_num=seg_num

        self.freq_len=input_len//2+1
        self.predf_len=pred_len//2+1
        freq_index_begin=dict()
        freq_index_end=dict()
        #seg_num:0/1/2……
        for i in range(seg_num,seg_num+1):
            seg=self.input_len//(2**i)
            freq_begin=[j*seg for j in range(2**i)]
            freq_end=[j*seg+seg for j in range(2**i)]
            freq_end[2**i-1]=input_len
            freq_index_begin[i]=freq_begin
            freq_index_end[i]=freq_end
        self.mask=dict()
        for i in range(seg_num,seg_num+1):
            themask=dict()
            for j in range(2**i):
                themask[j]=torch.zeros(self.input_len)
                themask[j][freq_index_begin[i][j]:freq_index_end[i][j]]=1
            self.mask[i]=themask

                    
        #self.Tweight = nn.Parameter(1/(2**(seg_num+1)-1)*torch.ones(2**(seg_num+1)-1, self.freq_len)).to(device)
        self.Tweight = nn.Parameter(1/(2**(seg_num))*torch.ones(2**(seg_num), self.freq_len)).to(device)
        #self.Freq_linear = nn.Linear(self.freq_len,self.predf_len).to(torch.cfloat)
    
        self.time_linear=nn.Sequential(nn.Linear(self.input_len,self.pred_len),
                                    nn.ReLU(),
                                    nn.Linear(self.pred_len,self.pred_len)  )

    def forward(self,x):

        x=x.unsqueeze(2)  #(B,D,1,L)
        segx=torch.zeros((x.shape[0],x.shape[1],2**self.seg_num,x.shape[3])).to(x.device)
        
        for i in range(self.seg_num,self.seg_num+1):
            for j in range(2**i):
                mask=self.mask[i][j]
                tmask=mask.to(x.device)
                nowx=x*tmask
                segx[:,:,j:j+1,:]=nowx
                
        segw = torch.fft.rfft(segx)#(B,D,S,L//2+1)
        

        FTweight = torch.softmax(self.Tweight, dim=0)  

        segw=segw*FTweight
        segw=torch.sum(segw,dim=2)#(B,D,L//2+1)

        
        y = torch.fft.irfft(segw, n=self.input_len)

        y=self.time_linear(y)

        return y 