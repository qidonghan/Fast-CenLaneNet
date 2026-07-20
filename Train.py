import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import utils.centernet as cennet
import model_zoo.LaneNet as LaneNet
import os
import Configs.config as cfg
from torch.utils.data import DataLoader
from utils.loss import CalCosinLoss
from Data.New_DataLoader import LaneDataset, LaneDataset_Test
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
torch.backends.cudnn.deterministic = True

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    B, C, H, W = inputs.shape
    assert C == 2, "inputs must have shape [B, 2, H, W] for binary segmentation"
    probs = F.softmax(inputs, dim=1)[:, 1, :, :]
    target = target.float()
    probs = probs.view(B, -1)
    target = target.view(B, -1)
    tp = torch.sum(probs * target, dim=1)
    fp = torch.sum(probs, dim=1) - tp
    fn = torch.sum(target, dim=1) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    loss = 1 - torch.mean(score)
    return loss

def TraintheNetwork(config):
    if not os.path.exists(config.savepath):
        os.mkdir(config.savepath)
    if config.Continue and os.path.exists(config.pretrained_model+'tempmodel.pkl'):
        model = LaneNet.LaneNet(config.The_selected_backbone,config)
        try:
            model.load_state_dict(torch.load(config.pretrained_model+'tempmodel.pkl'))
        except:
            temp=torch.load(config.pretrained_model+'tempmodel.pkl',map_location="cuda:0")
            from collections import OrderedDict
            new_state_dict=OrderedDict()
            for k,v in temp.items():
                name=k[7:]
                new_state_dict[name]=v
            model.load_state_dict(new_state_dict)
        print("load...")
    else:
         print("This will Train from scratch")
         model = LaneNet.LaneNet(config.The_selected_backbone,config)
    model.cuda()
    model.train()
    optimizer=optim.Adam(model.parameters(),lr=cfg.training_cfg.base_lr)
    Dataset=LaneDataset(config.GtDataroot)
    VDataset=LaneDataset_Test(config.GtDataroot)
    MyDataloader=DataLoader(Dataset,batch_size=config.batchsize,shuffle=True,num_workers=4,pin_memory=True)
    MyVDataloader=DataLoader(VDataset,batch_size=config.batchsize,shuffle=True,num_workers=4,pin_memory=True)
    lr_sheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        div_factor=20,
        final_div_factor=1e4,
        pct_start=0.1,
        anneal_strategy='cos',
        steps_per_epoch=len(MyDataloader),
        epochs=config.max_iter,
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95
    )
    all_lrs = []
    all_moms = []
    minloss=1000000
    LOSS=[]
    validation_LOSS=[]
    for iteration in range(0,config.max_iter):
        total_loss=[]
        model.train()
        for data in tqdm(MyDataloader):
            input=data[0].float().cuda()
            instancegt=data[1].long().cuda()
            label = data[2].long().cuda()
            cenlabel=data[3].float().cuda()
            output = model.forward(input)
            segres = output[1]
            cen_seg = output[2]
            insres = output[0]
            insres=insres/insres.norm(dim=1).unsqueeze(1)
            class_weights = torch.tensor([0.2, 1.0]).cuda()
            loss_ce = F.cross_entropy(segres, label, weight=class_weights, reduction='mean')
            loss_dice = Dice_loss(segres, label)
            loss1 = 0.5 * loss_ce + 0.5 * loss_dice
            loss2 = cennet._neg_loss(cen_seg, cenlabel)
            loss3 = CalCosinLoss(insres, instancegt, margin=config.instance_Margin, alpha=5000)
            loss =config.seg_cen_instance[0] * loss1 + config.seg_cen_instance[1] * loss2 + config.seg_cen_instance[2] * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(float(loss.detach()))
            lr_sheduler.step()
            pg = optimizer.param_groups[0]
            if 'momentum' in pg:
                cur_mom = pg['momentum']
            else:
                cur_mom = pg['betas'][0]
            all_lrs.append(lr_sheduler.get_last_lr()[0])
            all_moms.append(cur_mom)
        vtotal_loss = []
        with torch.no_grad():
            model.eval()
            for data in tqdm(MyVDataloader):
                input = data[0].float().cuda()
                instancegt = data[1].long().cuda()
                label = data[2].long().cuda()
                cenlabel = data[3].float().cuda()
                output = model.forward(input)
                segres = output[1]
                cen_seg = output[2]
                insres = output[0]
                insres = insres / insres.norm(dim=1).unsqueeze(1)
                class_weights = torch.tensor([0.2, 1.0]).cuda()
                loss_ce = F.cross_entropy(segres, label, weight=class_weights, reduction='mean')
                loss_dice = Dice_loss(segres, label)
                loss1 = 0.5 * loss_ce + 0.5 * loss_dice
                loss2 = cennet._neg_loss(cen_seg, cenlabel)
                loss3 = CalCosinLoss(insres, instancegt, margin=config.instance_Margin, alpha=5000)
                loss = config.seg_cen_instance[0] * loss1 + config.seg_cen_instance[1] * loss2 + config.seg_cen_instance[2] * loss3
                vtotal_loss.append(float(loss.detach()))
        if iteration>=0:
            seg_res = output[1]
            seg_res = seg_res[0, :, :, :].cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
            out = seg_res
            out = np.argmax(out, axis=2)
            cen_seg_res = output[2]
            cen_seg_res = cen_seg_res[0, :, :, :].cpu().squeeze(0).detach().numpy()
            ins_res = insres[0, :, :, :].cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
            ins_res[out == 0, :] = 0
            plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(out)
            plt.subplot(1, 3, 2)
            plt.imshow(cen_seg_res)
            plt.subplot(1, 3, 3)
            plt.imshow(ins_res[:,:,0:3])
            plt.pause(0.01)
            LOSS.append(np.stack(total_loss).mean())
            validation_LOSS.append(np.stack(vtotal_loss).mean())
            if np.stack(vtotal_loss).mean()<minloss:
                minloss=np.stack(vtotal_loss).mean()
                torch.save(model.state_dict(), config.savepath + 'besttempmodel.pkl')
                print("bestmodel_selected")
            plt.figure("loss")
            plt.clf()
            plt.plot(np.array(range(0,len(LOSS))),np.stack(LOSS),'-',c='b',label='Trainloss')
            plt.plot(np.array(range(0, len(validation_LOSS))), np.stack(validation_LOSS), '-', c='r', label='validationloss')
            plt.legend()
            plt.title("loss")
            plt.pause(0.01)
            plt.savefig('loss.png')
        torch.save(model.state_dict(),config.savepath+'tempmodel.pkl')
        if iteration%config.saveinterval==0:
            torch.save(model.state_dict(), config.savepath + 'iter_'+str(iteration)+'_tempmodel.pkl')
    plt.figure(2, figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(all_lrs)
    plt.title("Learning Rate")
    plt.xlabel("Batch step")
    plt.subplot(1,2,2)
    plt.plot(all_moms)
    plt.title("Momentum")
    plt.xlabel("Batch step")
    plt.tight_layout()
    plt.savefig("lr_momentum_curve.png", dpi=300)
    plt.close(2)

if __name__ == '__main__':
    print("here")
    parser=argparse.ArgumentParser()
    parser.add_argument("--max_iter",default=100,type=int)
    parser.add_argument("--batchsize",default=5,type=int)
    parser.add_argument("--Continue",default=0,type=int)
    parser.add_argument("--pretrained_model",default='./tempmodel/')
    parser.add_argument("--savepath",default='./tempmodel/')
    parser.add_argument("--instance_Margin",default=0,type=float)
    parser.add_argument("--saveinterval",default=10,type=int)
    parser.add_argument("--seg_cen_instance",default=[50,0.05,0.0001])
    parser.add_argument("--The_selected_backbone",default='SlimResNet',choices=['SlimResNet'])
    parser.add_argument("--GtDataroot",default="/path/to/tusimple/train_set/",type=str)
    parser.add_argument('--device',default=1,type=str)
    config=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    print(torch.cuda.get_device_name(0))
    TraintheNetwork(config)
