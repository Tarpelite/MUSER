from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset
import csv
import librosa
import pytorch_lightning as pl
import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as AT
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch_audiomentations import Compose, Gain, Shift, AddColoredNoise ,PolarityInversion,LowPassFilter
#from torchaudio_augmentations import *

IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

torchaudio.set_audio_backend("sox_io")

class GtzanDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str="/path/gtzan/",max_audio_length=480000, batch_size:int=32):
        super().__init__()
        self.data_dir= data_dir
        self.max_audio_length = max_audio_length
        self.batch_size = batch_size
        self.data = GtzanData(data_dir=self.data_dir, 
                                max_audio_length=self.max_audio_length,
                                set_name='train_filtered.txt',
                                train=True)
                                
        self.val_data = GtzanData(data_dir=self.data_dir, 
                                max_audio_length=self.max_audio_length,
                                set_name='valid_filtered.txt',
                                train=False) 
                                
        self.test_data = GtzanData(data_dir=self.data_dir, 
                                max_audio_length=self.max_audio_length,
                                set_name='test_filtered.txt',
                                train=False) 
        
        
    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self):   
        return DataLoader(self.val_data, batch_size=self.batch_size , num_workers=8, shuffle=False)
    
    def test_dataloader(self):   
        return DataLoader(self.test_data, batch_size=1 , num_workers=8, shuffle=False)

class GtzanData(Dataset):
    def __init__(self, data_dir, max_audio_length=480000, set_name=None, train=True):
        self.data_dir = data_dir
        self.set_name = set_name  
        self.max_audio_length = max_audio_length
        self.data_path = os.path.join(self.data_dir, self.set_name)
        self.train_data = pd.read_csv(self.data_path)
        self.sr = 16000
        self.train = train
        # data aug
        
        self.aug_transform = Compose(
            transforms=[
                AddColoredNoise(min_snr_in_db=0.0, max_snr_in_db= 20.0),
                Shift(p=0.4),
                Gain(min_gain_in_db=-15.0,max_gain_in_db=5.0, p=0.2),
                PolarityInversion(p=0.8),
                LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=2000,p=0.5),

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
        return len(self.train_data)
    
        
    def __getitem__(self, index):
        example = self.train_data.loc[index]
        all_genres = ['classical','blues','country','disco','hiphop','jazz','metal','pop','reggae','rock'] 
        genre = example.music_name.split('/')[0]
        labels = all_genres.index(genre)
        
        audio_name = example.music_name     
        path = os.path.join(self.data_dir, audio_name)  

        audio_torch, sr = torchaudio.load(path)
        
        

        if audio_torch.shape[0]> 1:
            audio_torch = audio_torch.mean(dim=0).unsqueeze(0)

        resampler = AT.Resample(sr, self.sr, dtype=audio_torch.dtype)
        
        resample_audio_torch = resampler(audio_torch)
        resample_audio_torch = resample_audio_torch[:, :self.max_audio_length]
        audio = F.pad(resample_audio_torch, (0, self.max_audio_length - resample_audio_torch.shape[1]), "constant", value=0)

        # do data aug
        if self.train:
            audio = self.aug_transform(audio.unsqueeze(0), sample_rate=self.sr)   
            audio = audio.squeeze(0)
            
        spec = self.spec_transform(audio)
        image_feature = self.spec_img_transform(spec.repeat(3, 1, 1))

        return audio, image_feature, labels
         
        
        



        
        



