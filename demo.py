import torch
from model.muser import MUSER

from model.audio_encoder.stft_encoder import STFTResNetWithAttention,STFTResNeXtWithAttention
from model.audio_encoder.wavelet_encoder import WaveletResNetWithAttention, WaveletResNeXtWithAttention



model = MUSER(pretrained=f'/path/muser/pt_weights/MUSER.pt')
#print(model)

audio_input = torch.randn(2,160000)
spec_input = torch.randn(2,3,224,224)
text = ["The song belongs to jazz","The song belongs to blue"]
text_input = [[label] for label in text] 


((audio_fea, spec_fea, text_fea), logits), loss = model(audio=audio_input,spec=spec_input, text=text_input)
print('audio_fea',audio_fea.shape)
print('spec_fea',spec_fea.shape)
print('text_fea',text_fea.shape)
print('logits',logits)
print('loss',loss)
