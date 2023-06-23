from resnet18 import ResNet18
from processed_dataset import STL10Pair
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from tqdm import tqdm
import argparse
import logging
import os
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self, resnet, out_dim):
        super().__init__()
        self.resnet = resnet
        self.linear = nn.Linear(512, out_dim)
    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x

def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def get_name(args):
    args_dict = vars(args)
    name = ""
    for key, value in args_dict.items():
        name += str(key) + "_" + str(value) + "#"
    return name

def check_and_mkdir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", type=int, help="resnet output dimension", default=256)
    parser.add_argument("-device", type=str, help="device of model and data", default="cuda:0")
    parser.add_argument("-batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-epoch", type=int, help="total epoch for self supervised training", default=500)
    parser.add_argument("-t", type=float, help="temperature", default=0.5)
    parser.add_argument("-train_num", type=int, help="training number", default=10000)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Net(resnet=ResNet18(), out_dim=args.out)
    model = model.to(device)
    log_root = "./log"
    tensorboard_log_root = "./tensorboard_log"
    model_pt_root = "./self-supervised_training_model_pt"
    check_and_mkdir(log_root)
    check_and_mkdir(tensorboard_log_root)
    check_and_mkdir(model_pt_root)

    name = get_name(args)
    log_name = name + ".log"
    log_path = os.path.join(log_root, log_name)
    tensorboard_log_path = os.path.join(tensorboard_log_root, name)
    check_and_mkdir(tensorboard_log_path)
    model_pt_path = os.path.join(model_pt_root, name)
    check_and_mkdir(model_pt_path)

    logger = init_logging(log_path)
    writer = SummaryWriter(tensorboard_log_path)

    dataset = STL10Pair(root="./data/STL10", download=False, train_num=args.train_num)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

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
            feat_norm = torch.norm(feat, dim=-1, keepdim=True)
            feat_norm_matrix = torch.matmul(feat_norm, feat_norm.t())
            sim = torch.exp(torch.matmul(feat, feat.t().contiguous()) / feat_norm_matrix / args.t)
            mask = (torch.ones_like(sim) - torch.eye(2 * batch_size, device=device)).bool()
            sim = sim.masked_select(mask).view(2 * batch_size, -1)
            feat1_norm = torch.norm(feat1, dim=-1)
            feat2_norm = torch.norm(feat2, dim=-1)
            pos_sim = torch.exp(torch.sum(feat1 * feat2, dim=-1) / (feat1_norm * feat2_norm) / args.t)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim.sum(dim=-1))).mean()
            epoch_loss += loss.item() * batch_size
            total_num += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch {epoch}, loss: {epoch_loss / total_num:.3f}")
        writer.add_scalar("self-supervised training loss", epoch_loss / total_num, epoch)

        if ((epoch + 1) % 50 == 0):
            logger.info(f"epoch {epoch}, saving checkpoint...")
            resnet_model_saved_path = os.path.join(model_pt_path, f"checkpoint_{epoch+1}.pt")
            torch.save(model.resnet.state_dict(), resnet_model_saved_path)
            logger.info("saving done.")
            







    