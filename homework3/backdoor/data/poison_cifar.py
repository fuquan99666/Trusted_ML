import os
import numpy as np
from copy import deepcopy
from torch import poisson_nll_loss
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image


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


def generate_trigger(trigger_type):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]] ### Define trigger as a 3x3 pattern
        ####################################
        ###write your code here return pattern(images with the trigger) and mask(indicating whether this poison needs to attach the trigger or not)
        ###Triggers add at the right bottom corner
        #### 2 points
        ####################################

        # we use checkerboard pattern as the trigger, 
        # so just set the pixel value of the right bottom corner with the trigger value and set the corresponding mask value to 1.
        pattern[29:32, 29:32, 0] = trigger_value 
        mask[29:32, 29:32, 0] = 1

        return pattern, mask
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]### Define trigger as a 3x3 pattern
        ####################################
        ###write your code here return pattern(images with the trigger) and mask(indicating whether this poison needs to attach the trigger or not)
        ###Triggers add at four corners
        #### 2 points
        ####################################

        # we use checkerboard_4corner pattern as the trigger,
        # so just set the pixel value of the four corners with the trigger value 
        # and set the corresponding mask value to 1.
        pattern[0:3, 0:3, 0] = trigger_value
        pattern[0:3, 29:32, 0] = trigger_value
        pattern[29:32, 0:3, 0] = trigger_value
        pattern[29:32, 29:32, 0] = trigger_value
        mask[0:3, 0:3, 0] = 1
        mask[0:3, 29:32, 0] = 1
        mask[29:32, 0:3, 0] = 1
        mask[29:32, 29:32, 0] = 1

        return pattern, mask

    elif trigger_type == 'gaussian_noise':
        ####################################
        ###write your code here return pattern(images with the trigger) and mask(indicating whether this poison needs to attach the trigger or not)
        ###Use image './data/cifar_gaussian_noise.png' as backdoor pattern. Trigger size 32*32
        #### 2 points
        ####################################

        # we use gaussian noise as the trigger, so just 
        # set all mask value to 1 and set the pattern value with the 
        # gaussian noise image value.
        pattern = np.array(Image.open('./data/cifar_gaussian_noise.png')).astype(np.uint8)
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)

        return pattern, mask
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def add_trigger_cifar(data_set, trigger_type, poison_rate, poison_target, trigger_alpha=1.0):
    """
    A simple implementation for backdoor attacks which only supports Badnets and Blend.
    :param clean_set: The original clean data.
    :param poison_type: Please choose on from [checkerboard_1corner | checkerboard_4corner | gaussian_noise].
    :param poison_rate: The injection rate of backdoor attacks.
    :param poison_target: The target label for backdoor attacks.
    :param trigger_alpha: The transparency of the backdoor trigger.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """

    # this function is used to clean-label attack, 
    # which add the trigger to the image but do not change the label.
    pattern, mask = generate_trigger(trigger_type=trigger_type)

    # for example, we want to attack the label 0, then we select all the samples whose 
    # label is not 0 to poison.
    poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
    poison_set = deepcopy(data_set)
    poison_num = int(poison_rate * len(poison_cand))
    choices = np.random.choice(poison_cand, poison_num, replace=False)

    for idx in choices:
        #### Add triggers to selected clean images to produce backdoor images (modify poison_set.data for selected sample)
        #### Modify poison images' labels (modify poison_set.targets for selected sample)
        #### write your code here
        #### Return a modified poison_set
        #### 2points
        #########################################
        
        # for these selected samples, we should 
        # add the trigger to the image and change the label to the target label.
        # and notice to use trigger_alpha to control the transparency of the trigger.
        mask = mask.astype(np.float32) * trigger_alpha
        pattern = poison_set.data[idx] * (1 - mask) + mask * pattern.astype(np.float32)
        poison_set.data[idx] = pattern.astype(np.uint8)
        poison_set.targets[idx] = poison_target

    trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                    'trigger_alpha': trigger_alpha, 'poison_target': poison_target,
                    'data_index': choices}
    return poison_set, trigger_info


def add_predefined_trigger_cifar(data_set, trigger_info):
    """
    Poisoning dataset using a predefined trigger. (Use to generate a poisoned test dataset)
    This can be easily extended to various attacks as long as they provide trigger information for every sample.
    :param data_set: The original clean dataset.
    :param trigger_info: The information for predefined trigger.
    :param exclude_target: Whether to exclude samples that belongs to the target label.
    :return: A poisoned dataset
    """
    if trigger_info is None:
        return data_set
    poison_set = deepcopy(data_set)

    pattern = trigger_info['trigger_pattern']
    mask = trigger_info['trigger_mask']
    trigger_alpha = trigger_info['trigger_alpha']
    poison_target = trigger_info['poison_target']
    #### Add triggers to all clean images to produce backdoor images (modify poison_set.data for all sample)
    #### Modify poison images' labels (modify poison_set.targets for all sample)
    #### write your code here
    #### Remove the samples whose original labels equal to the target label
    #### Return a modified poison_set
    #### 2points
    #########################################
    # we want to make some poison test dataset, so
    # we should add the trigger to them , and change their label to 
    # the target label. And remove the samples whose original labels equal to the target label because they do not need to be attacked.
    mask = mask.astype(np.float32) * trigger_alpha
    
    for idx in range(len(poison_set.targets)):
        if poison_set.targets[idx] != poison_target:
            pattern = poison_set.data[idx] * (1 - mask) + mask * pattern.astype(np.float32)
            poison_set.data[idx] = pattern.astype(np.uint8)
            poison_set.targets[idx] = poison_target

        else:
            # remove it 
            poison_set.data[idx] = np.zeros_like(poison_set.data[idx])
            poison_set.targets[idx] = -1

    return poison_set


class CIFAR10CLB(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(CIFAR10CLB, self).__init__()
        self.data = np.load(os.path.join(root, 'data.npy')).astype(np.uint8)
        self.targets = np.load(os.path.join(root, 'label.npy')).astype(np.long)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    clean_set = CIFAR10(root='../../data')
    poison_set, _ = add_trigger_cifar(data_set=clean_set, trigger_type='checkerboard_1corner', poison_rate=1.0, poison_target=0)
    import matplotlib.pyplot as plt
    print(poison_set.__getitem__(0))
    x, y = poison_set.__getitem__(0)
    plt.imshow(x)
    plt.show()


