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
from dataloader_gtzan import GtzanDataModule
import warnings
import os
import torchmetrics
import torchvision as tv
from model.muser import MUSER



logger = pl_loggers.TensorBoardLogger('tb_logs', name='music-clip')
warnings.filterwarnings("ignore", category=UserWarning)
pl.seed_everything(2021)





class MusicCLIP(pl.LightningModule):
    def __init__(self, lr=2e-5, epoch=100, batch_size=8):
        super().__init__()
        
        self.lr = lr
        self.batch = batch_size
        self.epoch = epoch
        self.model = MUSER(pretrained=f'/home/renjiawei/share_project_xy/muser/assets/MUSER.pt')
        self.fc = nn.Linear(1024,10)
        self.genre_label =  ['classical','blues','country','disco','hiphop','jazz','metal','pop','reggae','rock'] 
        self.accurcy = torchmetrics.Accuracy()
        self.template = "The song belongs to {}"

        
        
    def training_step(self, batch, batch_idx):
        #print('start training:')
        audio_input, img_input, label_input = batch
      
        text = []
        for i in range(label_input.shape[0]):
            genre = self.genre_label[label_input[i].item()]
            temp = self.template.format(genre)
            text.append(temp)
            
        text_input = [[label] for label in text]      
        _, loss = self.model(audio=audio_input, text=text_input, image=img_input)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) 
        return loss    
          

    def validation_step(self, batch, batch_idx):
        #print('start validating:')
        audio_input, img_input, label_input = batch

        text_input = [[self.template.format(label)] for label in self.genre_label]
        
        
        audio_features = self.model.encode_audio(audio_input)
        text_features = self.model.encode_text(text_input)
        #_, loss = self.model(audio_input, img_input, text_input)
        
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)  #torch.Size([batchsize, batchsize])
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)  #torch.Size([num_class, 1024])

        scale_audio_text = torch.clamp(self.model.logit_scale_at.exp(), min=1.0, max=100.0)

        logits_audio_text = scale_audio_text * audio_features @ text_features.T   #torch.Size([batchsize, num_class])
        pred_audio = logits_audio_text.argmax(dim=-1)

        return {"pred_audio": pred_audio, "label_input": label_input}
            
    def validation_epoch_end(self, outputs) -> None:

        all_outputs = self.all_gather(outputs)
        # print(len(all_outputs))
        # print([x["pred_audio"].shape for x in all_outputs])

        all_pred_audio = torch.cat([output["pred_audio"].view(-1) for output in all_outputs], dim=0)
        all_label_input = torch.cat([output["label_input"].view(-1) for output in all_outputs], dim=0)
        self.accurcy.reset()
        acc = self.accurcy(all_pred_audio, all_label_input)

        if self.trainer.global_rank == 0:
            print()
            print("acc:{}".format(acc))
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    
    def test_step(self, batch, batch_idx):
        audio_input, img_input, label_input = batch

        text_input = [[self.template.format(label)] for label in self.genre_label]
        
        
        audio_features = self.model.encode_audio(audio_input)
        text_features = self.model.encode_text(text_input)
        #_, loss = self.model(audio_input, img_input, text_input)
        
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)  #torch.Size([batchsize, batchsize])
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)  #torch.Size([num_class, 1024])

        scale_audio_text = torch.clamp(self.model.logit_scale_at.exp(), min=1.0, max=100.0)

        logits_audio_text = scale_audio_text * audio_features @ text_features.T   #torch.Size([batchsize, num_class])
        pred_audio = logits_audio_text.argmax(dim=-1)

        return {"pred_audio": pred_audio, "label_input": label_input}
    
    
    def test_epoch_end(self, outputs) -> None:

        all_outputs = self.all_gather(outputs)
        # print(len(all_outputs))
        # print([x["pred_audio"].shape for x in all_outputs])

        all_pred_audio = torch.cat([output["pred_audio"].view(-1) for output in all_outputs], dim=0)
        all_label_input = torch.cat([output["label_input"].view(-1) for output in all_outputs], dim=0)
        self.accurcy.reset()
        acc = self.accurcy(all_pred_audio, all_label_input)

        if self.trainer.global_rank == 0:
            print()
            print("acc:{}".format(acc))
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    
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
    parser.add_argument("--checkpoint_path", type=str, default=None)
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    dm = GtzanDataModule(
        data_dir=args.data_dir,
        max_audio_length=args.max_audio_length,
        batch_size=args.batch_size,      
    )

    #dm.setup("fit")

    print("Now Building model ...")
    
    model = MusicCLIP(lr=args.learning_rate, epoch=args.max_epochs, batch_size=args.batch_size)
    
    if args.checkpoint_path:
        print("loading model from ckpt path {}".format(args.checkpoint_path))
        model = MusicCLIP(lr=args.learning_rate, epoch=args.max_epochs, batch_size=args.batch_size).load_from_checkpoint(args.checkpoint_path)
    
    trainer = pl.Trainer.from_argparse_args(args)
    
    
    checkpoint_callback = ModelCheckpoint(monitor="val_acc_epoch", save_top_k=3, mode="max", filename="{epoch:02d}-{loss:.3f}-{val_acc_epoch:.3f}")
    bar = ProgressBar()
    bar._trainer = trainer
    trainer.callbacks = [checkpoint_callback, bar]
    
    
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(args.output_dir, "model_gtzan.bin"))

    print('testing:')
    trainer.test()
    

    