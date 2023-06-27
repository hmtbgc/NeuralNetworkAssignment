import torch
import torch.nn as nn
import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from resnet import ResNet18
from torch.optim import Adam, SGD
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self, resnet, num_class, pretrained_path):
        super().__init__()
        self.resnet = resnet
        self.fc = nn.Linear(512, num_class)
        self.resnet.load_state_dict(torch.load(pretrained_path))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-pretrained", type=str, help="path of pretrained resnet")
    parser.add_argument("-device", type=str, help="device of model and data", default="cuda:1")
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-epoch", type=int, help="total epoch of training linear part", default=100)
    args = parser.parse_args()
    
    linear_evaluation_log_root = "./linear_evaluation_log" 
    linear_evaluation_tensorboard_root = "./linear_evaluation_tensorboard_log"
    model_pt_root = "./linear_evaluation_model_pt"
    check_and_mkdir(linear_evaluation_log_root)
    check_and_mkdir(linear_evaluation_tensorboard_root)
    check_and_mkdir(model_pt_root)
    temp = args.pretrained.split("/")
    pretrained_file_name = temp[-2] + "_" + temp[-1].strip(".pt")
    log_name = get_now() + get_name(args) + "-" + pretrained_file_name
    logger = init_logging(os.path.join(linear_evaluation_log_root, log_name + ".log"))
    writer = SummaryWriter(os.path.join(linear_evaluation_tensorboard_root, log_name))
    model_pt_path = os.path.join(model_pt_root, log_name)
    check_and_mkdir(model_pt_path)
    

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(32),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    # ])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    
    trainset = CIFAR10(root="./data/cifar10", train=True, transform=transform_train, download=False)
    testset = CIFAR10(root="./data/cifar10", train=False, transform=transform_test, download=False)

    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = Net(ResNet18(), 10, args.pretrained)
    model = model.to(device)
    for param in model.resnet.parameters():
        param.requires_grad = False
        
    optimizer = SGD(model.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    loss_function = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        total_number = 0
        for data, target in tqdm(train_dataloader):
            batch_size = data.shape[0]
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            loss = loss_function(out, target)
            epoch_loss += loss.item() * batch_size
            total_number += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch {epoch}, loss:{epoch_loss / total_number:.3f}")
        writer.add_scalar("train loss", epoch_loss / total_number, epoch)
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], epoch)

        if ((epoch + 1) % 5 == 0):
            logger.info("############################## testing... ############################")
            torch.save(model.state_dict(), os.path.join(model_pt_path, f"checkpoint_{epoch+1}.pt"))
            model.eval()
            with torch.no_grad():
                total_number = 0
                correct_number = 0
                for data, target in tqdm(test_dataloader):
                    batch_size = data.shape[0]
                    data = data.to(device)
                    out = model(data)
                    pred = torch.argmax(out, dim=-1)
                    pred = pred.detach().cpu()
                    correct_number += (pred == target).sum()
                    total_number += batch_size
                acc = correct_number / total_number
                logger.info(f"epoch {epoch}, test accuracy: {acc * 100:.2f}%")
                writer.add_scalar("test top1 accuracy", acc, epoch)
                if (best_test_acc < acc):
                    best_test_acc = acc
                    torch.save(model.state_dict(), os.path.join(model_pt_path, "best.pt"))
            logger.info("################################# testing done ####################################")
                
        scheduler.step()







