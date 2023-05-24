from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
import os
import logging
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import argparse
import configs
from model import VGG16, ResNet18, MobileNetV2

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def eval(dataloader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        top1_correct = 0
        top5_correct = 0
        tot = 0
        loss_tot = 0.0
        for batch in tqdm(dataloader):
            data, label = batch
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = loss_fn(out, label)
            loss_tot += loss.item()
            out = F.softmax(out, dim=-1).detach().cpu().numpy()
            label = label.cpu().numpy()
            top1_preds = np.argmax(out, axis=-1)
            top1_correct += np.sum(label == top1_preds)
            tot += label.shape[0]
            top5_preds = np.argsort(out, axis=1)[:, -5:]
            top5_correct += np.sum([label[i] in top5_preds[i] for i in range(label.shape[0])])
        acc = top1_correct / tot
        top1_err = 1.0 - acc
        top5_err = 1.0 - top5_correct / tot
        
    return loss_tot / len(dataloader), acc, top1_err, top5_err


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type", default="vgg16")
    args = parser.parse_args()
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(configs.CIFAR100_TRAIN_MEAN, configs.CIFAR100_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(configs.CIFAR100_TRAIN_MEAN, configs.CIFAR100_TRAIN_STD)
    ])

    Train_dataset = CIFAR100(root="./data", train=True, download=False, transform=transform_train)
    train_number = int(len(Train_dataset) * configs.train_rate)
    train_indices = list(range(len(Train_dataset)))
    train_idx, valid_idx = train_indices[:train_number], train_indices[train_number:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_dataset = Subset(Train_dataset, train_idx)
    valid_dataset = Subset(Train_dataset, valid_idx)
    test_dataset = CIFAR100(root="./data", train=False, download=False, transform=transform_test)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.test_batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.net == "vgg16":
        model = VGG16(num_class=configs.num_class).to(device)
        test_model = VGG16(num_class=configs.num_class).to(device)
        model_name = "VGG16"
    elif args.net == "resnet18":
        model = ResNet18(num_class=configs.num_class).to(device)
        test_model = ResNet18(num_class=configs.num_class).to(device)
        model_name = "ResNet18"
    elif args.net == "mobilenetv2":
        model = MobileNetV2(num_class=configs.num_class).to(device)
        test_model = MobileNetV2(num_class=configs.num_class).to(device)
        model_name = "MobileNetV2"
    else:
        raise NotImplementedError(f"{args.net} is not implemented! Please try vgg16/resnet18/mobilenetv2")
    model_pt_root = configs.model_pt_saved_root
    data_argument_method = "baseline"
    model_pt_root = os.path.join(model_pt_root, data_argument_method)
    if not os.path.exists(model_pt_root):
        os.mkdir(model_pt_root)
    log_path = os.path.join(configs.log_path, data_argument_method)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tensorboard_log_path = os.path.join(configs.tensorboard_log_path, data_argument_method)
    if not os.path.exists(tensorboard_log_path):
        os.mkdir(tensorboard_log_path)
    model_pt_path = os.path.join(model_pt_root, f"{model_name}.pt")
    logger = init_logging(os.path.join(configs.log_path, f"{model_name}.log"))
    writer = SummaryWriter(os.path.join(configs.tensorboard_log_path, f"{model_name}"))
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=configs.initial_lr, momentum=0.9, weight_decay=5e-4)
    warmup_scheduler = WarmUpLR(optimizer, configs.warmup_epoch * len(train_dataloader))
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.2)
    
    best_valid_acc = 0.0

    for epoch in range(configs.Epoch):
        model.train()
        epoch_loss = 0.0
        if (epoch > configs.warmup_epoch):
            train_scheduler.step(epoch)
        for batch in tqdm(train_dataloader):
            data, label = batch
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            if (epoch <= configs.warmup_epoch):
                warmup_scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch: {epoch}, loss:{epoch_loss / len(train_dataloader):.4f}")
        logger.info(f"Epoch: {epoch}, loss:{epoch_loss / len(train_dataloader):.4f}")
        writer.add_scalar("Train loss", epoch_loss / len(train_dataloader), epoch)
        if ((epoch + 1) % configs.eval_every == 0):
            valid_loss, acc, top1_err, top5_err = eval(valid_dataloader, model, loss_fn, device)              
            if (acc > best_valid_acc):
                best_valid_acc = acc
                torch.save(model.state_dict(), os.path.join(model_pt_root, f"{model_name}.pt"))
            print(f"Epoch: {epoch}, valid accuracy:{acc * 100:.2f}%, top1 error:{top1_err * 100:.2f}%, top5 error:{top5_err * 100:.2f}%, best valid accuracy:{best_valid_acc * 100:.2f}%")
            logger.info(f"Epoch: {epoch}, valid accuracy:{acc * 100:.2f}%, top1 error:{top1_err * 100:.2f}%, top5 error:{top5_err * 100:.2f}%, best valid accuracy:{best_valid_acc * 100:.2f}%")
            writer.add_scalar("Valid accuaray", acc, epoch)
            writer.add_scalar("Valid top1 error", top1_err, epoch)
            writer.add_scalar("Valid top5 error", top5_err, epoch)
            writer.add_scalar("Valid loss", valid_loss, epoch)
            
    # testing...
    test_model.load_state_dict(torch.load(model_pt_path))
    _, test_acc, test_top1_err, test_top5_err = eval(test_dataloader, test_model, loss_fn, device)
    print(f"test accuracy: {test_acc * 100:.2f}%, test top1 error: {test_top1_err * 100:.2f}%, test top5 error:{test_top5_err * 100:.2f}%")
    logger.info(f"test accuracy: {test_acc * 100:.2f}%, test top1 error: {test_top1_err * 100:.2f}%, test top5 error:{test_top5_err * 100:.2f}%")
        
                

        