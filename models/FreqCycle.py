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
        self.seg_window=configs.seg_window
        self.seg_stride=configs.seg_stride
        self.seg_num = (self.seq_len - self.seg_window) // self.seg_stride + 1

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        self.Tweight = nn.Parameter(1/(self.seg_num)*torch.ones(self.seg_num, self.seq_len//2+1),requires_grad=True)
        


        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
        

        self.seg_layer=Time_seg(self.seq_len,self.pred_len,self.seg_window,self.seg_stride,configs.d_model,configs.window_type,configs.gpu)
        #print(list(self.seg_layer.named_parameters()))
        self.Cycfilter=CycFilter(self.pred_len,self.enc_in,configs.gpu)
        '''abalation study
        self.MLP=nn.Sequential(
                nn.Linear(self.seq_len,  self.pred_len),
                nn.ReLU(),
                nn.Linear( self.pred_len, self.pred_len)
            )
        '''

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data(FECF)
        x_period=self.cycleQueue(cycle_index, self.seq_len)
        x = x - x_period

        # forecasting with SFPL

        y=self.seg_layer(x.permute(0, 2, 1),self.Tweight.to(x.device)).permute(0, 2, 1)
        
        #abalation study
        #y=self.MLP(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y_period=self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        #y_period=self.Cycfilter(y_period)
        y = y + y_period

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y



''' Use s to divide x in to 2**s segments
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
        self.Tweight = nn.Parameter(1/(2**(seg_num))*torch.ones(2**(seg_num), self.freq_len),requires_grad=True).to(device)
        #self.Freq_linear = nn.Linear(self.freq_len,self.predf_len).to(torch.cfloat)
    
        self.time_linear=nn.Sequential(nn.Linear(self.input_len,self.pred_len),
                                        nn.ReLU(),
                                        nn.Linear(self.pred_len,self.pred_len)  )
        #self.compconv=ComplexConv2D(2**(seg_num))
    def forward(self,x,Tweight):

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
        #FTweight = Tweight
        FTweight = torch.softmax(Tweight, dim=0) 
        #print(self.Tweight.grad)
        #print(self.Tweight[0,:])
        ########Low Pass Filter 
        #segw[:,:,:,20:]=0

        segw=segw*FTweight
        segw=torch.sum(segw,dim=2)*2**self.seg_num#(B,D,L//2+1)

        
        y = torch.fft.irfft(segw, n=self.input_len)

        y=self.time_linear(y)

        return y 
'''
#Use sliding window to split x
class Time_seg(nn.Module):
    def __init__(self, input_len, pred_len, window_size, stride,d_model,window_type, device):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.window_size = window_size
        self.stride = stride
        #window_type 'rect', 'hamming', 'hann', or 'gaussian'
        gaussian_sigma = 0.4  # Only used for Gaussian window

        
        # Calculate number of segments
        self.seg_num = (input_len - window_size) // stride + 1
        self.freq_len = input_len // 2 + 1
        self.predf_len = pred_len // 2 + 1
        
        # Create masks and windows for each segment
        self.mask = dict()
        self.windows = dict()  # Store window functions
        
        for i in range(self.seg_num):
            start = i * stride
            end = start + window_size
            
            # Create rectangular mask
            mask = torch.zeros(self.input_len)
            mask[start:end] = 1
            self.mask[i] = mask
            
            # Create window functions
            if window_type == 'hamming':
                window = torch.hamming_window(window_size, periodic=False, device=device)
            elif window_type == 'hann':
                window = torch.hann_window(window_size, periodic=False, device=device)
            elif window_type == 'gaussian':
                # Create Gaussian window
                x = torch.linspace(-3, 3, window_size, device=device)
                window = torch.exp(-x**2 / (2 * gaussian_sigma**2))
                window = window / window.max()  # Normalize to [0,1]

            else:  # rectangular
                window = torch.ones(window_size, device=device)
                
            # Store window with zero-padding to match input length
            full_window = torch.zeros(self.input_len, device=device)
            full_window[start:end] = window
            self.windows[i] = full_window
        
        self.time_linear = nn.Sequential(
            nn.Linear(self.input_len, d_model),
            nn.ReLU(),
            nn.Linear( d_model, self.pred_len)
        )
        
    def forward(self, x, Tweight):
        x = x.unsqueeze(2)  # (B, D, 1, L)
        segx = torch.zeros((x.shape[0], x.shape[1], self.seg_num, x.shape[3]), device=x.device)
        
        # Apply windowed segmentation
        for i in range(self.seg_num):
            window = self.windows[i].to(x.device)
            nowx = x * window  # Apply window function
            segx[:, :, i:i+1, :] = nowx
            
        # Compute FFT for each segment
        segw = torch.fft.rfft(segx)  # (B, D, S, L//2+1)
        
        # Apply learned weights
        FTweight = torch.softmax(Tweight, dim=0)
        segw = segw * FTweight
        segw = torch.sum(segw, dim=2)  # (B, D, L//2+1)
        
        # Inverse FFT and time projection
        y = torch.fft.irfft(segw, n=self.input_len)
        y = self.time_linear(y)
        
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

