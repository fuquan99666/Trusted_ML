import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Parameters you cannot change
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.') ## (1-alpha)*ori_img+alpha
## 1.0 for badnets and clean-label
## 0.2 for blend

# Basic model parameters. You can change

parser.add_argument('--batch-size', type=int, default=256, help='the batch size for dataloader')
# backdoor parameters. You can change
parser.add_argument('--clb-dir', type=str, default='./data/clean-label/0.1/')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign'], help='type of backdoor attacks used during training')
args = parser.parse_args()
os.makedirs('output', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join('output', 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    # Load Data 
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create poisoned / clean dataset
    orig_train = CIFAR10(root='data', train=True, download=True, transform=transform_train)
    '''Split original Training set into to parts:
    1. clean_train: In attack, we use it to generate.
    2. clean_defense: In defense stage, we use it to generate backdoor triggers.
    ''' 

    # just split the original training set into two parts.
    clean_train, clean_defense = poison.split_dataset(dataset=orig_train, val_frac=0.1,
                                                  perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
   
    clean_test = CIFAR10(root='data', train=False, download=True, transform=transform_test)
    
    # These are some triggers used for badnets, clean-label, and blend way.
    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None}
    
    # select the trigger type according to the argument.
    trigger_type = triggers[args.poison_type]
    
    if args.poison_type in ['badnets', 'blend']:
        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=0.05,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    elif args.poison_type == 'clean-label':
        ## Clean-Label Attack
        poison_train = poison.CIFAR10CLB(root=args.clb_dir, transform=transform_train)
        pattern, mask = poison.generate_trigger(trigger_type=triggers['clean-label'])
        trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                        'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    elif args.poison_type == 'benign':
        ## Natural Training
        poison_train = clean_train
        poison_test = clean_test
        trigger_info = None
    else:
        raise ValueError('Please use valid backdoor attacks: [badnets | blend | clean-label]')

    poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, 'resnet18')(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join('output', 'model_init.th'))
    if trigger_info is not None:
        torch.save(trigger_info, os.path.join('output', 'trigger_info.th'))
    for epoch in range(1, 50):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                data_loader=poison_train_loader)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

    torch.save(net.state_dict(), os.path.join('output', str(args.poison_type)+'model_last.th'))


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            if len(labels.shape) == 2:
                labels = labels.squeeze(1)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()
