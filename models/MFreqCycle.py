import torch
import torch.nn as nn

class RecurrentCycle(torch.nn.Module):
    
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.seg_num=configs.seg_num
        self.Cycle_num=configs.Cycle_num
        self.Cycle_stride=configs.Cycle_stride
        self.Cycle_window=configs.Cycle_window
        self.every_Cycle=configs.every_Cycle
        self.recent_num=configs.recent_num


        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)


        self.seg_layer=Time_seg(self.seq_len,self.pred_len,self.seg_num,configs.gpu)
        self.Cycfilter=CycFilter(self.pred_len,self.enc_in,configs.gpu)
        
        self.cycleQueue = nn.ModuleList()
        self.seg_layer = nn.ModuleList()
        self.Cycfilter = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.up_pool = nn.ModuleList()
        self.dropout=nn.ModuleList()
        if self.Cycle_num>=1:
            for i in range(self.Cycle_num):
                if self.Cycle_window[i]>1:
                    self.pooling.append(torch.nn.AvgPool1d(self.Cycle_window[i],stride=self.Cycle_stride[i]))
                    self.up_pool.append(nn.Linear((self.pred_len-self.Cycle_window[i])//self.Cycle_stride[i]+1,self.pred_len,bias=True))
                    self.seg_layer.append(Time_seg((self.recent_num[i]-self.Cycle_window[i])//self.Cycle_stride[i]+1,(self.pred_len-self.Cycle_window[i])//self.Cycle_stride[i]+1,self.seg_num,configs.gpu))
                    self.Cycfilter.append(CycFilter((self.pred_len-self.Cycle_window[i])//self.Cycle_stride[i]+1,self.enc_in,configs.gpu))
                    self.cycleQueue.append(RecurrentCycle(cycle_len=self.every_Cycle[i], channel_size=self.enc_in))
                    self.dropout.append(nn.Dropout(0.1))
                else:
                    self.seg_layer.append(Time_seg(self.recent_num[i],self.pred_len,self.seg_num,configs.gpu))
                    self.Cycfilter.append(CycFilter(self.pred_len,self.enc_in,configs.gpu))
                    self.cycleQueue.append(RecurrentCycle(cycle_len=self.every_Cycle[i], channel_size=self.enc_in))
                    self.dropout.append(nn.Dropout(0.1))



    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        y_sum=torch.zeros(x.shape[0],self.pred_len,self.enc_in).to(x.device)

        for i in range(self.Cycle_num-1):
            x_down=x[:,-self.recent_num[i]:,:]
            if self.Cycle_window[i]>1:
                x_down=self.pooling[i](x_down.permute(0,2,1)).permute(0,2,1)

            x_cycle=self.cycleQueue[i](cycle_index[:,i],x_down.size(1))

            x_down=x_down-x_cycle
            y_down=self.seg_layer[i](x_down.permute(0, 2, 1)).permute(0, 2, 1)
            y_period=self.cycleQueue[i](((cycle_index[:,i]+self.recent_num[i])//self.Cycle_stride[i])%(self.every_Cycle[i]//self.Cycle_stride[i]),(self.pred_len-self.Cycle_window[i])//self.Cycle_stride[i]+1)
            y_period=self.Cycfilter[i](y_period)
            y=y_down+y_period
            y=self.dropout[i](y)
            if self.Cycle_window[i]>1:
                y=self.up_pool[i](y.permute(0,2,1)).permute(0,2,1)
            y_sum=y+y_sum





        # instance denorm
        if self.use_revin:
            y_sum = y_sum * torch.sqrt(seq_var) + seq_mean

        return y_sum




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
        #self.compconv=ComplexConv2D(2**(seg_num))
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
        
        ########
        FTweight = torch.softmax(self.Tweight, dim=0) 

        segw=segw*FTweight
        segw=torch.sum(segw,dim=2)#(B,D,L//2+1)

        
        y = torch.fft.irfft(segw, n=self.input_len)

        y=self.time_linear(y)

        return y 
    

class CycFilter(nn.Module):
    def __init__(self,pred_len,enc_in,gpu):
        super().__init__()
        self.pred_len=pred_len
        self.predf_len=pred_len//2+1
        self.enc_in=enc_in
        self.Fweight= nn.Parameter(torch.ones(self.predf_len)).to(gpu)
    def forward(self,x):
        x=x.permute(0,2,1)
        fx=torch.fft.rfft(x)

        fx=fx*self.Fweight
        y = torch.fft.irfft(fx, n=self.pred_len)
        y=y.permute(0,2,1)
        return y
