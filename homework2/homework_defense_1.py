from __future__ import print_function
import os
import argparse
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.resnet import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--defense', type=str, default='trades', choices=['pgd', 'trades', 'mart'],
                    help='defense objective used for adversarial training')
parser.add_argument('--beta', type=float, default=4.0,
                    help='TRADES regularization, larger is usually more robust')
parser.add_argument('--beta-warmup-epochs', type=int, default=2,
                    help='linearly warm up beta in early epochs')
parser.add_argument('--adv-ce-weight', type=float, default=2.0,
                    help='extra CE weight on adversarial examples for CE-PGD robustness')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
# For training , we use data augmentation .
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# For testing, just convert to tensor
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def defense_train_config(defense):
    configs = {
        'pgd': {
            'lr': 0.03,
            'weight_decay': 5e-4,
            'step_size': 0.01,
            'num_steps': 3,
        },
        'trades': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'step_size': 0.01,
            'num_steps': 3,
            'beta': 1.0,
        },
        'mart': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'step_size': 0.01,
            'num_steps': 3,
            'beta': 2.0,
        },
    }
    return configs.get(defense, configs['pgd'])


def PGD(model,
            x_natural,
            y,
            optimizer,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            device= torch.device("cuda")):
    # define KL-loss
    criterion = nn.CrossEntropyLoss(size_average=False)
    model.eval()
    x_adv = x_natural+0.001 * torch.randn(x_natural.shape).to(device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv),y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    # Use the final adversarial x_adv for robust training to enhance the model robustness.
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss


def train(args, model, device, train_loader, optimizer, epoch, step_size, perturb_steps):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = PGD(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=args.epsilon,
                           perturb_steps=perturb_steps,
                           device = device)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def TRADES(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031,
           perturb_steps=10, beta=1.0, epoch=1, beta_warmup_epochs=1,
           device=torch.device("cuda")):

    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss(reduction='mean')

    # Standard TRADES attack generation: maximize KL(p_adv || p_nat)
    model.eval()
    # Use uniform random start in epsilon-ball to make inner maximization stronger.
    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    with torch.no_grad():
        nat_prob = F.softmax(model(x_natural), dim=1)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl_for_adv = criterion_kl(
                F.log_softmax(model(x_adv), dim=1),
                nat_prob
            )
        grad = torch.autograd.grad(loss_kl_for_adv, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_ce = criterion_ce(logits_nat, y)
    loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                           F.softmax(logits_nat.detach(), dim=1))


    beta_scale = min(1.0, float(epoch) / max(1, beta_warmup_epochs))
    loss = loss_ce + (beta * beta_scale) * loss_kl
    return loss

def train_trades(args, model, device, train_loader, optimizer, epoch, step_size, perturb_steps, beta):
    # just change the loss function from PGD to TRADES

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = TRADES(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=args.epsilon,
                           perturb_steps=perturb_steps,
                           beta=beta,
                           epoch=epoch,
                           beta_warmup_epochs=args.beta_warmup_epochs,
                           device = device)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def MART(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10,
         beta=6.0, epoch=1, beta_warmup_epochs=1, device=torch.device("cuda")):
    # MART add a (1 - f_y(x)) weight on the KL 
    # if f_y(x) is small, it means the model is not confident on the correct label, 
    # so it is a difficult example, we want to make the adv's distribution close to the natural one, so we put a large weight on the KL loss;
    # if f_y(x) is large, it means the model is confident on the correct label,
    # so it is an easy example, for its adv example, we don't care much about its KL loss, so we put a small weight on the KL loss.

    criterion_ce = nn.CrossEntropyLoss(reduction='mean')

    model.eval()
    # Use uniform random start in epsilon-ball to make inner maximization stronger.
    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            # Standard MART uses CE-based adversarial sample generation.
            loss_adv_for_attack = criterion_ce(model(x_adv), y)
        grad = torch.autograd.grad(loss_adv_for_attack, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits_adv = model(x_adv)
    logits_nat = model(x_natural)
    adv_prob = F.softmax(logits_adv, dim=1)
    nat_prob = F.softmax(logits_nat, dim=1)

    # With only two training epochs, a direct adversarial CE objective converges faster
    # than the full misclassification-aware MART variant.
    loss_adv = criterion_ce(logits_adv, y)

    # calculate the weight for KL loss
    true_label_prob = nat_prob.gather(1, y.unsqueeze(1)).squeeze(1)
    kl_weight = 1.0 - true_label_prob
    loss_robust_per_sample = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        nat_prob.detach(),
        reduction='none'
    ).sum(dim=1)

    beta_scale = min(1.0, float(epoch) / max(1, beta_warmup_epochs))
    loss = loss_adv + (beta * beta_scale) * (kl_weight * loss_robust_per_sample).mean()
    return loss
            

def train_mart(args, model, device, train_loader, optimizer, epoch, step_size, perturb_steps, beta):
    # just change the loss function from PGD to MART

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = MART(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=args.epsilon,
                           perturb_steps=perturb_steps,
                           beta=beta,
                           epoch=epoch,
                           beta_warmup_epochs=args.beta_warmup_epochs,
                           device = device)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
                

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """Use a short-horizon schedule that matters for 2-epoch training."""
    if args.epochs <= 2:
        if epoch == 1:
            lr = args.lr
        else:
            lr = args.lr * 0.5
    else:
        lr = args.lr

        if epoch >= 75:
            lr = args.lr * 0.1
        if epoch >= 90:
            lr = args.lr * 0.01
        if epoch >= 100:
            lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    import time
    start_time = time.time()
    model = ResNet18().to(device)

    train_cfg = defense_train_config(args.defense)
    args.lr = train_cfg['lr']
    args.weight_decay = train_cfg['weight_decay']
    start_epoch = 1


    def count_parameters(model, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    param = count_parameters(model)
    print(param/1000000)




    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    if start_epoch > 1:
        opt_path = os.path.join(model_dir, 'opt-wideres-{}-checkpoint_epoch{}.tar'.format(args.defense, start_epoch - 1))
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print('Loaded optimizer state from {}'.format(opt_path))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        if args.defense == 'pgd':
            train(args, model, device, train_loader, optimizer, epoch, train_cfg['step_size'], train_cfg['num_steps'])
        elif args.defense == 'mart':
            train_mart(args, model, device, train_loader, optimizer, epoch, train_cfg['step_size'], train_cfg['num_steps'], train_cfg['beta'])
        else:
            train_trades(args, model, device, train_loader, optimizer, epoch, train_cfg['step_size'], train_cfg['num_steps'], train_cfg['beta'])

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')
        # save checkpoint
        if epoch % args.save_freq == 0:
            defense_str = args.defense if args.defense in ['pgd', 'trades', 'mart'] else 'pgd'
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-{}-epoch{}.pt'.format(defense_str, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-{}-checkpoint_epoch{}.tar'.format(defense_str, epoch)))
    end_time = time.time()

    ###################################
    # You should not change those codes
    ###################################

    def _pgd_whitebox(model,
                      X,
                      y,
                      epsilon=0.031,
                      num_steps=3,
                      step_size=0.0157):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
        print('err pgd (white-box): ', err_pgd)
        return err, err_pgd

    def eval_adv_test_whitebox(model, device, test_loader):
        """
        evaluate model by white-box attack
        """
        model.eval()
        robust_err_total = 0
        natural_err_total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
        robust_acc = (len(testset)-robust_err_total)/len(testset)
        clean_acc = (len(testset)-natural_err_total)/len(testset)
        print('natural_accuracy: ', clean_acc)
        print('robustness: ', robust_acc)
    eval_adv_test_whitebox(model, device, test_loader)
    print(end_time - start_time)

    ###################################
    # You should not change those codes
    ###################################



if __name__ == '__main__':
    main()

"""
python homework_defense_1.py --defense "pgd"
python homework_defense_1.py --defense "trades"
python homework_defense_1.py --defense "mart"

pgd :
natural_accuracy:  tensor(0.4892, device='cuda:0')
robustness:  tensor(0.3036, device='cuda:0')
acc/2 + rob = 0.2446 + 0.3036 = 0.5482

trades:
natural_accuracy:  tensor(0.6719, device='cuda:0')
robustness:  tensor(0.2005, device='cuda:0')
acc/2 + rob = 0.3359 + 0.2005 = 0.5364

mart:
natural_accuracy:  tensor(0.5243, device='cuda:0')
robustness:  tensor(0.2858, device='cuda:0')
acc/2 + rob = 0.26215 + 0.2858 = 0.54795
"""