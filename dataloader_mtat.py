from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset
import csv
import librosa
import pytorch_lightning as pl
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as AT
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch_audiomentations import Compose, Gain, Shift, AddColoredNoise ,PolarityInversion, LowPassFilter

IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

torchaudio.set_audio_backend("sox_io")



class MTATDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str='/path/tagging/split/mtat',max_audio_length=480000, batch_size:int=100):
        super().__init__()
        self.data_dir= data_dir
        self.max_audio_length = max_audio_length
        self.batch_size = batch_size 
        
        self.train_data = MTATData(self.data_dir, max_audio_length=self.max_audio_length, set_name="train.npy", train=True) 
        self.val_data = MTATData(self.data_dir, max_audio_length=self.max_audio_length, set_name="valid.npy", train=False)
        self.test_data = MTATData(self.data_dir, max_audio_length=self.max_audio_length, set_name="test.npy", train=False) 
       
       
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=12, shuffle=True)
   
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=12, shuffle=False)
        
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=12, shuffle=False)    
 


class MTATData(Dataset):
    def __init__(self, data_dir ,max_audio_length=480000, set_name=None, train=True):
        self.data_dir = data_dir     
        self.set_name = set_name           # set_name = "train.npy","valid.npy","test.npy"
        self.max_audio_length = max_audio_length
        self.data_path = os.path.join(self.data_dir, self.set_name)      

        self.data = np.load(self.data_path)
        self.labels = np.load('/path/tagging/split/mtat/binary.npy')
        self.sr = 16000
        self.train = train

        # data aug
        self.aug_transform = Compose(
            transforms=[
                AddColoredNoise(min_snr_in_db=5.0, max_snr_in_db= 20.0),
                #Shift(p=0.5),
                Gain(min_gain_in_db=-5.0,max_gain_in_db=5.0, p=0.5),
                #PolarityInversion(p=0.5),
                LowPassFilter(min_cutoff_freq=400, max_cutoff_freq=1000,p=0.5),

            ]
        )
        

        self.spec_transform = AT.Spectrogram(
            n_fft=1024,
            win_length=None,
            hop_length=512,
            center=True,
            pad_mode='reflect',
            power=2.0,
        )



        self.spec_img_transform = tv.transforms.Compose([
            #     tv.transforms.ToTensor(),
                tv.transforms.Resize(300, interpolation=Image.BICUBIC),
                tv.transforms.CenterCrop(224),
                tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
            ])
        
        
        
    def __len__(self):
        return len(self.data)


    def convert(self,name):
        path = '/path/tagging/npy_stft'
        npy_path = os.path.join(path, name.split('/')[1][:-3]) + 'npy'
        return npy_path
        
    # input :  'a/janine_johnson-chopin_recital-01-waltz_kk_iva_no_12_in_e_major-0-29.mp3' 
    # output: '/path/tagging/npy_stft/janine_johnson-chopin_recital-01-waltz_kk_iva_no_12_in_e_major-0-29.npy'
    
    def __getitem__(self, index):       
        id, name = self.data[index].split('\t')     #id = '2583'  #name = 'a/janine_johnson-chopin_recital-01-waltz_kk_iva_no_12_in_e_major-0-29.mp3'
        id = int(id)
        label = self.labels[id]
        
        wav_path = os.path.join('/path/tagging', name)
        
        # print(wav_path)
        
        audio_torch, sr = torchaudio.load(wav_path)
        

        if audio_torch.shape[0]> 1:
            audio_torch = audio_torch.mean(dim=0).unsqueeze(0)
        
        resampler = AT.Resample(sr, self.sr, dtype=audio_torch.dtype)
        
        resample_audio_torch = resampler(audio_torch)
        resample_audio_torch = resample_audio_torch[:, :self.max_audio_length]
        audio = F.pad(resample_audio_torch, (0, self.max_audio_length - resample_audio_torch.shape[1]), "constant", value=0)
        
        # img_path = self.convert(name)
        # img_feature = np.load(img_path)
        # img_feature = torch.from_numpy(img_feature)  #torch[1,224,224]
        # print(img_feature.shape)

        if self.train:
            # do data aug
            audio = self.aug_transform(audio.unsqueeze(0), sample_rate=self.sr)
            audio = audio.squeeze(0)
        
        spec = self.spec_transform(audio)
        image_feature = self.spec_img_transform(spec.repeat(3, 1, 1))
        # print(image_feature.shape)
        
        return audio, image_feature, label
        
        
        



        
        



