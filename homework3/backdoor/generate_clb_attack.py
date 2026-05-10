import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
import models



def split_dataset(dataset, val_frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    :param val_frac: the fraction of validation set.
    :param perm: A predefined permutation for sampling. If perm is None, generate one.
    :return: A training set + a validation set
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_val = int(val_frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_val:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_val:]].tolist()

    # generate the test set
    val_set = deepcopy(dataset)
    val_set.data = val_set.data[perm[:nb_val]]
    val_set.targets = np.array(val_set.targets)[perm[:nb_val]].tolist()
    return train_set, val_set

def generate_trigger():
    pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
    mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
    trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]

    ####################################
    ###write your code here return pattern(images with the trigger) and mask(indicating whether this poison needs to attach the trigger or not)
    ###Triggers add at four corners
    ###Same as poison cifar, therefore no points 
    ####################################

    return pattern, mask


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, max_attack_iters, restarts):
    """
    : model: target model for the adversarial attack
    : X: input images
    : y: input labels
    : epsilon: maximum perturbation budget
    : alpha: step_size for each pgd iteration
    : max_attack_iters: maximum pgd iteration for each input images
    : restarts: you need to run (restarts+1) times pgd attacks to get the worst pertubation for each images
    """
    y = y.unsqueeze(dim=0)
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    loss_fc = torch.nn.CrossEntropyLoss()
    for _ in range(restarts+1):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)   #restart with random initialized delta

        ######### Using PGD Attack with restarthere to generate hard examples ####
        # Additional Requirements: Update perturb images only if they can be correctly classified.
        # For example, if image x[1]+delta[1] can be corretly calssified while image x[2]+delta[2] cannot, only update delta[1].
        # Restart: regenerate delta and only use delta with the maximum loss
        # Return max delta (worst pertubation with the maximum loss for each input images after multiple restarts)
        # Please your code here 
        # 2 Points

        # each x should have its own delta
        for _ in range(max_attack_iters):
            delta.requires_grad = True
            output = model(X + delta)

            # check if image can be correctly classified, 
            # and we only update that part.
            correct_mask = (output.argmax(dim=1) == y).float().unsqueeze(dim=1)
            
            loss = loss_fc(output, y)
            grad = torch.autograd.grad(loss, [delta])[0]
            delta = delta.detach() + correct_mask * alpha * torch.sign(grad.detach())
            delta = clamp(delta, -epsilon, epsilon)
        
        if loss > max_loss:
            max_loss = loss
            max_delta = delta.detach()
        
    return max_delta


transform_train = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])
orig_train = CIFAR10(root="data", train=True, download=True, transform=transform_train)
clean_test = CIFAR10(root="data", train=False, download=True, transform=transform_test)
clean_train, clean_val = split_dataset(dataset=orig_train, val_frac=0.05,
                                                    perm=np.loadtxt('data/cifar_shuffle.txt', dtype=int))
train_loader = DataLoader(clean_train, batch_size=1, num_workers=0)
test_loader = DataLoader(clean_test, batch_size=64, num_workers=0)
# Load model
model = getattr(models, 'resnet18')(num_classes=10)
checkpoint = torch.load("benignmodel_last.th")
model.load_state_dict(checkpoint)
model = nn.DataParallel(model.cuda())
model.eval()
model.cuda()

criterion = torch.nn.CrossEntropyLoss()
poison_rate = 0.1
epsilon = 16 / 255.
alpha = 0.24 / 255.

clean_train_x = list()
clean_train_y = list()

for step, (X, y) in tqdm.tqdm(enumerate(train_loader)):
    X, y = X.cuda(), y.cuda()
    clean_train_x.append(X)
    clean_train_y.append(y.to('cpu', torch.uint8).numpy())

clean_train_y = np.concatenate(clean_train_y, axis=0)
pattern, mask = generate_trigger()
poison_cand = [i for i in range(len(clean_train_y)) if clean_train_y[i] == 0]
poison_num = int(poison_rate * len(poison_cand))
choices = np.random.choice(poison_cand, poison_num, replace=False)
backdoor_list = np.zeros(clean_train_y.shape[0])

for idx in tqdm.tqdm(choices):
    pgd_delta = attack_pgd(model, clean_train_x[idx], torch.from_numpy(np.array(int(clean_train_y[idx]))).cuda(),
                           epsilon, alpha, 100, 1)
    clean_train_x[idx] = clean_train_x[idx] + pgd_delta

for i, image in tqdm.tqdm(enumerate(clean_train_x)):
    clean_train_x[i] = image.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
clean_train_x = np.concatenate(clean_train_x, axis=0)


for idx in tqdm.tqdm(choices):
    orig = clean_train_x[idx]
    ### Attach Trigger to the image
    clean_train_x[idx] = np.clip(
        (1 - mask) * orig + mask * pattern, 0, 255
    ).astype(np.uint8)
    ###############################
    x = Image.fromarray(clean_train_x[idx])
    backdoor_list[idx] = 1
    ###### output one image####
    x.save("image_cifar10_clean_label.png")
root = './data/clean-label/{:.1f}/'.format(poison_rate)
os.makedirs(root, exist_ok=True)
np.save(root + 'data.npy', clean_train_x)
np.save(root + 'label.npy', clean_train_y)
np.save(root + 'backdoor_list.npy', backdoor_list)