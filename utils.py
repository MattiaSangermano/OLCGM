
import torch
import copy
import matplotlib.image as mpimg
import json

from torchvision import transforms
from avalanche.evaluation.metric_results import MetricValue
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from os.path import expanduser
from model import ResNet18
from avalanche.models import SimpleMLP
from torchvision.utils import make_grid, save_image

from matplotlib import pyplot as plt
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import MinibatchLoss

def get_modified_dataset(dataset, num_examples=-1, val_size = 0):
    if dataset == 'mnist':
        train_set = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                      train=True, download=True)
        test_set = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                     train=False, download=False)
    elif dataset == 'cifar10':
        train_set = CIFAR10(root=expanduser("~") + "/.avalanche/data/cifar10/",
                      train=True, download=True)
        test_set = CIFAR10(root=expanduser("~") + "/.avalanche/data/cifar10/",
                     train=False, download=False)
    elif dataset == 'fashion':
        train_set = FashionMNIST(root=expanduser("~") + "/.avalanche/data/fashion/",
                      train=True, download=True)
        test_set = FashionMNIST(root=expanduser("~") + "/.avalanche/data/fashion/",
                     train=False, download=False)
    elif dataset == 'svhn':
        train_set = SVHN(root=expanduser("~") + "/.avalanche/data/svhn/",
                      split='train', download=True)
        test_set = SVHN(root=expanduser("~") + "/.avalanche/data/svhn/",
                     split='test', download=True)
    else:
        assert False, 'wrong dataset'

    all_data = {}
    data = {}

    if num_examples == -1:
        num_examples = len(train_set)
    if val_size > 0:
        num_examples += val_size
    indexes = []
    if dataset == 'svhn':
        targets = torch.Tensor(train_set.labels)
    else:
        targets = train_set.targets

    for index, el in enumerate(targets):
        if isinstance(el, torch.Tensor):
            el = el.item()
        if el not in all_data:
            all_data[el] = 1
            indexes.append(index)
        elif all_data[el] < num_examples:
            indexes.append(index)
            all_data[el] += 1
    indexes = torch.tensor(indexes)

    if dataset == 'svhn':
        train_set.labels = targets[indexes]
        train_set.data = train_set.data[indexes]
        train_set.targets = train_set.labels
        test_set.targets = test_set.labels
    else:
        if isinstance(targets, torch.Tensor):
            train_set.targets = torch.index_select(targets, 0, indexes)
            train_set.data = torch.index_select(train_set.data, 0, indexes)
        else:
            train_set.targets = torch.index_select(torch.tensor(targets, dtype=torch.long), 0, indexes).tolist()
            train_set.data = train_set.data[indexes]

    return train_set, test_set

def log_metrics(wandb_logger, loss, name):
        if wandb_logger is not None:
            wandb_logger.log_metric(
                    MetricValue(origin=MinibatchLoss(), name=name,
                                value=loss, x_plot=0),
                    '')

def save_images(dataset, mem_size, imgs, name, save_type='single', wandb_logger=None, iter=None,):
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        channel = 3
    elif dataset == 'mnist':
        mean = [0.1307]
        std = [0.3081]
        channel = 1
    elif dataset == 'fashion':
        mean = [0.2860]
        std = [0.3530]
        channel = 1
    elif dataset == 'svhn':
        mean = [0.4376821, 0.4437697, 0.47280442]
        std =  [0.19803012, 0.20101562, 0.19703614]
        channel = 3
    else:
        assert False, 'wrong dataset'

    image_syn_vis = copy.deepcopy(imgs.detach().cpu())
    for ch in range(channel):
        image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
    if save_type == 'single':
        for i in range(len(image_syn_vis)):
            filename = 'image_' + str(i) + '_' + str(iter) + '_' + name + '.png'
            img_to_plot = image_syn_vis[i].detach().data
            img_to_plot[img_to_plot < 0] = 0.0
            img_to_plot[img_to_plot > 1] = 1.0
            save_image(img_to_plot, './' + filename)
            if wandb_logger is not None:
                image = transforms.ToPILImage()(img_to_plot)
                log_metrics(wandb_logger, image, filename)
            if False:
                img = mpimg.imread('./' + filename)
                imgplot = plt.imshow(img)
                plt.show()
    elif save_type == 'grid':
        filename = 'grid_' + name + '.png'
        grid = make_grid(image_syn_vis, mem_size // 10)
        save_image(grid, './' + filename)
        if wandb_logger is not None:
            filename = filename.replace('_', '/')
            image = transforms.ToPILImage()(grid)
            log_metrics(wandb_logger, image, filename)

def get_network(model_name, image_size):
    if model_name == 'mlp':
        input_size = image_size[0] * image_size[1] * image_size[2]
        model = SimpleMLP(num_classes=10, hidden_size=400,
                          input_size=input_size)
    elif model_name == 'resnet':
        model = ResNet18(nclasses=10,input_size=image_size,  nf=20)
    else:
        assert False, 'wrong model'
    return model

def get_statistics(weights_dict, masks, statistics, exp_counter):

    num_ones = 0
    num_zeros = 0
    num_imgs = 0
    for c, weights in weights_dict.items():
        weights = weights * masks[c]
        num_ones += len(weights[torch.max(weights, 1).values == 1.0])
        num_zeros += len(weights[torch.max(weights, 1).values == 0.0])
        num_imgs += len(weights)
    if exp_counter not in statistics:
        statistics[exp_counter] = {}

    statistics[exp_counter].update({
            'num_zeros': num_zeros,
            'num_ones': num_ones,
            'num_imgs': num_imgs
        })
    with open('statistics.json', 'w+') as f:
        json.dump(statistics, f, indent=4)

