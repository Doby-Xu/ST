import copy

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.models
from pytorch_lightning.utilities import rank_zero_only
import torchmetrics
from torch.utils.data import DataLoader
import model
from transformers import AdamW,get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
# from utils import load_for_transfer_learning
from models import *
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))

class deep_feature(torch.nn.Module):
    def __init__(self):
        super(deep_feature, self).__init__()
        self.model=InceptionResnetV1(pretrained='vggface2',classify=False)

    def forward(self,x):
        x=self.model(x)
        return x

class TrainWrapper(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1.5e-4,#2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 40,#0,
        predictions_file: str = 'predictions.pt',
        num_labels: int = 10,
        checkpoints: str  = './checkpoints/model.pth',
        pretrained_model:str='../checkpoints/T2t_vit_24_pretrained.pth.tar',
        k1:float=1.,
        k2:float=1.,
        first_keep_rate:float=1.,
        transform:object=None,
        transform_value = 0.8,
        transform_name:str='None',
        R = False,
        C = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.G = model.G()
        self.F = model.F(k1,k2, R = R)
        shuffle_flag = 'R' if R else 'None'
        checkpoint = torch.load("../checkpoint_celeba_timm_pretrain/ckpt_0.05_"+shuffle_flag + '_' + transform_name + '_' + str(transform_value)+".pth")
        self.F.load_state_dict(checkpoint['net'])
        
        # self.accuracy_metric = torchmetrics.Accuracy()
        self.criterion = torch.nn.MSELoss()#.CrossEntropyLoss()
        self.metric_ssim=SSIM(data_range=1., size_average=True, channel=3)
        self.psnr=PSNR()
        self.best = 1.
        self.checkpoints = checkpoints
        self.deep_feature=deep_feature()
        self.identify_net=InceptionResnetV1(pretrained='vggface2',classify=True,num_classes=10177)
        #checkpoint = torch.load("/home/liyaox/yaodixi/bishe/t2t/FaceIdentify/checkpoint_celeba_T2t_vit_24/ckpt_0.01_0.0005_98.46988385924905_noPos.pth")
        #self.identify_net.load_state_dict(checkpoint['net'])
        # self.identify_net = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=5749)
        # checkpoint = torch.load( "/home/liyaox/yaodixi/bishe/t2t/FaceIdentify/checkpoint_lfwa_T2t_vit_24/ckpt_0.01_0.0005_99.94710194211441.pth")
        # self.identify_net.load_state_dict(checkpoint['net'])
        self.transform=transform
        self.transform_value=transform_value
        self.transform_name=transform_name
        self.R = R

    def metric(self, preds, labels, mode='val'):
        a = self.criterion(preds, labels).item()  # self.accuracy_metric(preds, labels)
        return {f'{mode}_acc': a}

    # def forward(self, inputs):
    #     return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x,_ = batch
        label=x.clone().detach().requires_grad_(True)
        if self.transform is not None:
            x = self.transform.process(x)
        with torch.no_grad():
            intermediate=self.F(x)
        outputs = self.G(intermediate)
        loss = self.criterion(outputs,label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x,_ = batch
        inter = x
        if self.transform is not None:
            inter = self.transform.process(x)
        with torch.no_grad():
            intermediate = self.F(inter)
        outputs = self.G(intermediate)
        val_loss = self.criterion(outputs,x)
        preds = outputs
        metric_dict = self.metric(preds, x)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)
        return (metric_dict['val_acc'],preds.shape[0],x.cpu().clone().detach())

    def validation_epoch_end(self,val_outs):
        correct = 0
        count = 0
        for a,b,c in val_outs:
            correct += a*b
            count += b
        c = c.to(self.F.model.pos_embed.device)
        if (correct/count)<self.best:
            self.best = correct/count
            torch.save(self.G.state_dict(),self.checkpoints)
            print("Current best mse%.3f"%(correct/count))
        vis = torchvision.utils.make_grid(c, nrow=8, padding=1, normalize=True)
        vis = torchvision.transforms.ToPILImage()(vis)
        vis.save('org_celeb.png')
        intermediate = self.F(c)
        vis = self.G(intermediate)
        vis = torchvision.utils.make_grid(vis, nrow=8, padding=1, normalize=True)
        vis -=torch.min(vis)
        vis =vis/torch.max(vis)
        vis = torchvision.transforms.ToPILImage()(vis)
        vis.save('rec_exp_'+str(self.R)+'_'+self.transform_name+'_'+str(self.transform_value)+'.png')


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, identities = batch
        identities=identities-1
        intermediate = self.F(x)
        outputs = self.G(intermediate)
        val_loss = self.criterion(outputs, x)
        ssim= self.metric_ssim(outputs,x)
        psnr=self.psnr(outputs,x)
        input_deep_feature=self.deep_feature(x)
        output_deep_feature=self.deep_feature(outputs)
        # for i in range(output_deep_feature.shape[0]):
        #     print(torch.cosine_similarity(output_deep_feature[0],output_deep_feature[i],dim=0))
        # print('----------------------')
        feature_sim=torch.mean(torch.abs(torch.cosine_similarity(input_deep_feature,output_deep_feature,dim=1)))
        preds = outputs
        metric_dict = self.metric(preds, x, mode='test')
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        #self.log('test_loss', val_loss, prog_bar=True)
        self.log('SSIM', ssim, prog_bar=True)
        self.log('PSMR', psnr, prog_bar=True)
        self.log('F-SIM', feature_sim, prog_bar=True)
        outputs = self.identify_net(outputs)
        _, predicted = outputs.max(1)
        correct = predicted.eq(identities).sum().item()
        total = identities.size(0)
        self.log('ID acc', correct/total, prog_bar=True)

        return (val_loss.item(),preds.shape[0],x.cpu().clone().detach())

    def test_epoch_end(self, val_outs):
        correct = 0
        count = 0
        for a, b, c in val_outs:
            correct += a * b
            count += b
        for a, b, c in val_outs:
            break
        #(correct/count)
        c=c.to(self.F.model.pos_embed.device)
        vis = torchvision.utils.make_grid(c, nrow=8, padding=1, normalize=True)
        vis = torchvision.transforms.ToPILImage()(vis)
        vis.save('org_test.png')
        intermediate = self.F(c)
        vis = self.G(intermediate)
        torch.save(vis, 'blackboxresult_PS+.pth')
        vis = torchvision.utils.make_grid(vis, nrow=8, padding=1, normalize=True)
        vis = torchvision.transforms.ToPILImage()(vis)
        vis.save('rec_test'+str(self.R)+'_'+self.transform_name+'_'+str(self.transform_value)+'.png')

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.G.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.G.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,betas=(0.9,0.95))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.G.save_pretrained(self.hparams.save_dir)

if __name__=="__main__":
    net=InceptionResnetV1(classify=False)
    x=torch.randn((5,3,224,224))
    x=net(x)
    print(x)