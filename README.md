# FreqCycle
## Model Implementation

We propose FreqCycle, a novel framework integrating: (i) a Filter-Enhanced Cycle Forecasting (FECF) module to extract low-frequency features by explicitly learning shared periodic patterns in the time domain, and (ii) a Segmented Frequency-domain Pattern Learning (SFPL) module to enhance mid to high frequency energy proportion via learnable filters and adaptive weighting. The core implementation code of FreqCycle is available at:

```
models/FreqCycle.py
```

For the FECF part, To identify the relative position of each sample within the recurrent cycles, we need to generate cycle index (i.e., **_t_ mod _W_** ) additionally for each data sample. The code for this part is available at:

```
data_provider/data_loader.py
```
For the SFPL part, The specific implementation code of SFPL :
```python
class Time_seg(nn.Module):
```
We have two segmentation strategies for x:
Sliding window-based segmentation
Uniform segmentation into 2**s parts (where s is a given input)



## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n FreqCycle python=3.10
conda activate FreqCycle
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for FreqCycle can be obtained from previous works such as Autoformer and SCINet. 
Create a separate folder named ```./data``` and place all the CSV files in this directory. 

### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance:

```
bash scripts/FreqCycle/etth1.sh

bash scripts/MFreqCycle/etth2.sh
```







