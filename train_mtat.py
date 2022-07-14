from pytorch_lightning import profiler
from pytorch_lightning.accelerators import accelerator
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
import argparse
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning import loggers as pl_loggers
from dataloader_mtat import MTATDataModule
import warnings
import os
import torchmetrics
from sklearn import metrics
import torchvision as tv
from model.muser import MUSER


logger = pl_loggers.TensorBoardLogger('tb_logs', name='music-clip')
warnings.filterwarnings("ignore", category=UserWarning)


class MusicCLIP(pl.LightningModule):
    def __init__(self, lr=2e-5, batch_size=8, epoch=20):
        super().__init__()
        
        self.lr = lr
        self.batch = batch_size
        self.epoch = epoch
        self.model = MUSER(pretrained=f'/path/muser/assets/MUSER.pt')
        self.tag_label =  ['guitar','classical','slow','techno','strings','drums','electronic','rock','fast','piano','ambient','beat','violin','vocal','synth','female','indian','opera','male','singing','vocals','no vocals','harpsichord','loud','quiet','flute','woman','male vocal','no vocal','pop','soft','sitar','solo','man','classic','choir','voice','new age','dance','male voice','female vocal','beats','harp','cello','no voice','weird','country','metal','female voice','choral']
        #self.auroc = torchmetrics.AUROC(num_classes = 50, pos_label = None,average = 'micro')
        #self.ap = torchmetrics.AveragePrecision() 

        #self.template = "The song belongs to {}"
        #self.template = "tags for the music is {}"
        self.template = "the music is characterized by {}"
        

    def training_step(self, batch, batch_idx):
        #print('start training:')
        audio_input, img_input, label_input = batch
        label_input = label_input.cpu().numpy()
                
        text = []
        for i in range(label_input.shape[0]):
            tags = []
            for j in range(50):
                if label_input[i][j] == 1:
                    tag = self.tag_label[j]
                    tags.append(tag)
            tags = self.template.format(",".join(tags)) 
            text.append(tags)
        text_input = [[label] for label in text]     
        

        _, loss = self.model(audio=audio_input, text=text_input, image=img_input)

        
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        
        return loss      
    
    
    def validation_step(self, batch, batch_idx):
        #print('start validating:')
        audio_input, img_input, label_input = batch

        text_input = [[self.template.format(label)] for label in self.tag_label]
        
        audio_features = self.model.encode_audio(audio_input)
        text_features = self.model.encode_text(text_input)
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)  #torch.Size([batchsize, batchsize])
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)  #torch.Size([num_class, 1024])

        scale_audio_text = torch.clamp(self.model.logit_scale_at.exp(), min=1.0, max=100.0)

        logits_audio_text = scale_audio_text * audio_features @ text_features.T   #torch.Size([batchsize, num_class])

        return {'logits': logits_audio_text, 'labels':label_input.long()}



    def validation_epoch_end(self, outputs) -> None:
        
        all_outputs = self.all_gather(outputs)
        all_logits_audio = torch.cat([x["logits"].view(-1, x["logits"].size(-1)) for x in all_outputs], dim=0)
        all_labels_input = torch.cat([x["labels"].view(-1, x["labels"].size(-1)) for x in all_outputs], dim=0)
        
        
        all_logits_audio = all_logits_audio.cpu().detach().numpy()
        all_labels_input = all_labels_input.cpu().detach().numpy()
        
        try:
            roc_aucs  = metrics.roc_auc_score(all_labels_input, all_logits_audio, average='macro')
            pr_aucs = metrics.average_precision_score(all_labels_input, all_logits_audio, average='macro')
        except Exception as e:
            print(e)
            roc_aucs = 0.0
            pr_aucs = 0.0

        if self.trainer.global_rank == 0:
            print("roc_aucs:{}, pr_aucs:{}".format(roc_aucs, pr_aucs))
        self.log('val_roc_auc_epoch', roc_aucs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) 
        self.log('val_pr_auc_epoch', pr_aucs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)     
        
        
    def test_step(self, batch, batch_idx):

        #print('start validating:')
        audio_input, img_input, label_input = batch

        text_input = [[self.template.format(label)] for label in self.tag_label]
        
        audio_features = self.model.encode_audio(audio_input)
        text_features = self.model.encode_text(text_input)

        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)  #torch.Size([batchsize, batchsize])
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)  #torch.Size([num_class, 1024])

        scale_audio_text = torch.clamp(self.model.logit_scale_at.exp(), min=1.0, max=100.0)

        logits_audio_text = scale_audio_text * audio_features @ text_features.T   #torch.Size([batchsize, num_class])

        return {'logits': logits_audio_text, 'labels':label_input.long()}




    def test_epoch_end(self, outputs) -> None:
        
        all_outputs = self.all_gather(outputs)
        all_logits_audio = torch.cat([x["logits"].view(-1, x["logits"].size(-1)) for x in all_outputs], dim=0)
        all_labels_input = torch.cat([x["labels"].view(-1, x["labels"].size(-1)) for x in all_outputs], dim=0)
        
        
        all_logits_audio = all_logits_audio.cpu().detach().numpy()
        all_labels_input = all_labels_input.cpu().detach().numpy()
        
        try:
            roc_aucs  = metrics.roc_auc_score(all_labels_input, all_logits_audio, average='macro')
            pr_aucs = metrics.average_precision_score(all_labels_input, all_logits_audio, average='macro')
        except Exception as e:
            print(e)
            roc_aucs = 0.0
            pr_aucs = 0.0

        if self.trainer.global_rank == 0:
            print("test_roc_aucs:{}, test_pr_aucs:{}".format(roc_aucs, pr_aucs))
            
        self.log('test_roc_auc_epoch', roc_aucs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) 
        self.log('test_pr_auc_epoch', pr_aucs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)     
        

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0, nesterov=True )
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / 5 if epoch < 5 else 0.5 * (math.cos((epoch - 5) /(self.epoch - 5) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        
        return [optimizer], [lr_scheduler]
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_audio_length", type=int, default=480000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, required=True) 
    parser.add_argument("--seed", type=int, default=42)
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    dm = MTATDataModule(
        data_dir=args.data_dir,
        max_audio_length=args.max_audio_length,
        batch_size=args.batch_size,      
    )


    print("Now Building model ...")
    
    checkpoint_path = args.checkpoint_path
    model = MusicCLIP(lr=args.learning_rate, batch_size=args.batch_size, epoch=args.max_epochs).load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor="val_pr_auc_epoch", save_top_k=3, mode="max", filename="{step:02d}-{val_roc_auc_epoch:.3f}-{val_pr_auc_epoch:.3f}")
    bar = ProgressBar()
    bar._trainer = trainer
    trainer.callbacks = [checkpoint_callback, bar]
    
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(args.output_dir, "model_mtat.bin"))
    
    print('testing:')

    trainer.test()
    
    
    
