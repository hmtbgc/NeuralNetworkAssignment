from resnet import ResNet18
from processed_dataset import CIFAR10Pair
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from tqdm import tqdm
import argparse
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from utils import *

class Net(nn.Module):
    def __init__(self, resnet, out_dim):
        super().__init__()
        self.resnet = resnet
        self.g = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.g(x)
        return F.normalize(x, dim=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", type=int, help="resnet output dimension", default=256)
    parser.add_argument("-device", type=str, help="device of model and data", default="cuda:0")
    parser.add_argument("-batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-epoch", type=int, help="total epoch for self supervised training", default=500)
    parser.add_argument("-t", type=float, help="temperature", default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Net(resnet=ResNet18(), out_dim=args.out)
    model = model.to(device)
    log_root = "./self-supervised_log"
    tensorboard_log_root = "./self-supervised_tensorboard_log"
    model_pt_root = "./self-supervised_model_pt"
    check_and_mkdir(log_root)
    check_and_mkdir(tensorboard_log_root)
    check_and_mkdir(model_pt_root)

    name = get_now() + get_name(args)
    log_name = name + ".log"
    log_path = os.path.join(log_root, log_name)
    tensorboard_log_path = os.path.join(tensorboard_log_root, name)
    check_and_mkdir(tensorboard_log_path)
    model_pt_path = os.path.join(model_pt_root, name)
    check_and_mkdir(model_pt_path)

    logger = init_logging(log_path)
    writer = SummaryWriter(tensorboard_log_path)

    dataset = CIFAR10Pair(root="./data/cifar10", download=False, train=True)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0, last_epoch=-1)

    for epoch in range(args.epoch):
        epoch_loss = 0.0
        total_num = 0
        for data in tqdm(dataloader):
            img1, img2 = data
            batch_size = img1.shape[0]
            img1 = img1.to(device)
            img2 = img2.to(device)
            feat1 = model(img1)
            feat2 = model(img2)
            feat = torch.cat([feat1, feat2], dim=0)
            sim = torch.exp(torch.mm(feat, feat.t().contiguous()) / args.t)
            mask = (torch.ones_like(sim) - torch.eye(2 * batch_size, device=device)).bool()
            sim = sim.masked_select(mask).view(2 * batch_size, -1)
            pos_sim = torch.exp(torch.sum(feat1 * feat2, dim=-1) / args.t)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim.sum(dim=-1))).mean()
            epoch_loss += loss.item() * batch_size
            total_num += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch {epoch}, loss: {epoch_loss / total_num:.3f}")
        writer.add_scalar("training loss", epoch_loss / total_num, epoch)

        if ((epoch + 1) % 10 == 0):
            logger.info("################## saving model pt... ########################")
            logger.info(f"epoch {epoch}, saving checkpoint...")
            resnet_model_saved_path = os.path.join(model_pt_path, f"checkpoint_{epoch+1}.pt")
            torch.save(model.resnet.state_dict(), resnet_model_saved_path)
            logger.info("######################## saving done ###########################")
            







    