import torch
import random
import numpy as np
import argparse

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import nc_benchmark
from avalanche.training.strategies import Naive
from avalanche.training.plugins.replay import ReplayPlugin,\
    ClassBalancedStoragePolicy, RandomExemplarsSelectionStrategy
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import ExperienceAccuracy, StreamAccuracy,\
    ExperienceForgetting, StreamForgetting, MinibatchLoss

from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.logging.wandb_logger import WandBLogger
from torchvision import transforms
from torchvision.transforms import ToTensor

from Plugins.LCGM import LCGM
from Plugins.OLCGM import OLCGM
from Plugins.OnlineReplay import OnlineReplay

from utils import get_modified_dataset, get_network
from avalanche.benchmarks import data_incremental_benchmark, benchmark_with_validation_stream

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def run_experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiences = 5
    epochs = 1

    if args.seed >= 0:
        set_seed(args.seed)
    else:
        args.seed = None

    if args.dataset == 'mnist':
        train_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        test_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_size = (1, 28, 28)
        train, test = get_modified_dataset('mnist', num_examples=args.num_ex, val_size=args.val_size)

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        image_size = (3, 32, 32)
        train, test = get_modified_dataset('cifar10', num_examples=args.num_ex, val_size=args.val_size)
    elif args.dataset == 'fashion':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        image_size = (1, 28, 28)
        train, test = get_modified_dataset('fashion', num_examples=args.num_ex, val_size=args.val_size)
    elif args.dataset == 'svhn':
        train_transform = test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442),
                (0.19803012, 0.20101562, 0.19703614))
        ])
        image_size = (3, 32, 32)
        train, test = get_modified_dataset('svhn', num_examples=args.num_ex, val_size=args.val_size)
    else:
        assert False, 'wrong dataset'
    ordering = [i for i in range(10)]

    scenario = nc_benchmark(
            train_dataset=train,
            test_dataset=test,
            n_experiences=experiences,
            task_labels=False,
            seed=args.seed,
            fixed_class_order=ordering,
            train_transform=train_transform,
            eval_transform=test_transform)
    size_experiences = []
    print(scenario.classes_order)
    for i, step in enumerate(scenario.train_stream):
        size_experiences.append(len(step.dataset) // args.mb_size)
    size_experiences[0] -= 1
    for i in range(1, len(size_experiences)):
        size_experiences[i] += size_experiences[i - 1]

    if args.val_size > 0:
        scenario = benchmark_with_validation_stream(scenario, validation_size= 2 * args.val_size, shuffle=True)

    streamAccuracy = StreamAccuracy()
    streamForgetting = StreamForgetting()
    experienceAccuracy = ExperienceAccuracy()
    experienceForgetting = ExperienceForgetting()
    minibatchLoss = MinibatchLoss()

    loggers = []
    if args.logger == 1:
        loggers.append(InteractiveLogger())

    wandb_logger = None
    logger = None
    if args.run is not None and args.logger == 1:
        args.project = 'thesis'

        wandb_logger = WandBLogger(project_name='thesis', run_name=args.run,
                                   config=args)
        loggers.append(wandb_logger)

    model = get_network(args.model_name, image_size)

    condensation_args = {
            'lr_net': 0.1,
            'iteration': 1,
            'outer_loop': args.ol,
            'inner_loop': args.il,
            'image_size': image_size,
            'lr_w': args.lr_w,
            'condense_new_data': args.condense_nw,
            'dataset': args.dataset,
            'l2_w': args.l2_w,
            'debug': args.debug,
            'plugin': args.plugin
        }

    if args.plugin == 'lcgm' or  args.plugin == 'gm':
        lcgm = LCGM(mem_size=args.memory, wandb_logger=wandb_logger, **condensation_args)
        plugins = [lcgm]
    elif args.plugin == 'olcgm' or args.plugin == 'ogm':
        condensation_args['k'] = args.k
        scenario = data_incremental_benchmark(scenario, args.mb_size, True, True)
        olcgm = OLCGM(mem_size=args.memory, wandb_logger=wandb_logger, **condensation_args)
        plugins = [olcgm]
    elif args.plugin == 'rr':
        replayPlugin = ReplayPlugin(
            args.memory,
            storage_policy=ClassBalancedStoragePolicy(
                ext_mem={}, mem_size=args.memory, adaptive_size=True,
                selection_strategy=RandomExemplarsSelectionStrategy())
            )
        plugins = [replayPlugin]
    elif args.plugin == 'orr':
        scenario = data_incremental_benchmark(scenario, args.mb_size, True, True)
        onlineReplayPlugin = OnlineReplay(
            args.memory,
            storage_policy=ClassBalancedStoragePolicy(
                ext_mem={}, mem_size=args.memory, adaptive_size=True,
                selection_strategy=RandomExemplarsSelectionStrategy())
            )
        plugins = [onlineReplayPlugin]

    logger = EvaluationPlugin(streamAccuracy, streamForgetting,
                              experienceAccuracy, experienceForgetting,
                              minibatchLoss,
                              loggers=loggers,
                              benchmark=scenario)

    if args.plugin == 'olcgm' or args.plugin == 'ogm' or args.plugin == 'orr':
        logger.active = False

    sgd = SGD(model.parameters(), lr=condensation_args['lr_net'])

    cl_strategy = Naive(
        model, sgd, CrossEntropyLoss(), train_mb_size=args.mb_size,
        train_epochs=epochs, eval_mb_size=100, plugins=plugins,
        evaluator=logger, device=device
    )

    accuracies = {}

    for i, step in enumerate(scenario.train_stream):
        metrics = None
        cl_strategy.train(step, num_workers=0)

        if args.plugin == 'olcgm' or args.plugin == 'ogm' or args.plugin == 'orr':

            if i in size_experiences:
                logger.active = True
                if args.val_size > 0:
                    metrics = cl_strategy.eval(scenario.valid_stream,
                                               num_workers=0)
                else:
                    metrics = cl_strategy.eval(scenario.test_stream,
                                               num_workers=0)
                logger.active = False
        else:
            if args.val_size > 0:
                metrics = cl_strategy.eval(scenario.valid_stream,
                                           num_workers=0)
            else:
                metrics = cl_strategy.eval(scenario.test_stream,
                                           num_workers=0)
        if metrics is not None:
            for key, val in metrics.items():
                if key.find('Top1_Acc_Exp') != -1:
                    if key not in accuracies:
                        accuracies[key] = val
                    else:
                        if accuracies[key] < val:
                            accuracies[key] = val
    final_accuracy = streamAccuracy.result()[0]
    final_forgetting = streamForgetting.result()

    if args.plugin == 'olcgm' or args.plugin == 'ogm' or args.plugin == 'orr':

        logger.active = True
        if args.val_size > 0:
            metrics = cl_strategy.eval(scenario.valid_stream,
                                        num_workers=0)
        else:
            metrics = cl_strategy.eval(scenario.test_stream,
                                        num_workers=0)
        logger.active = False
        final_forgetting = 0
        for key, val in metrics.items():
            if key in accuracies:
                final_forgetting += accuracies[key] - val
        final_forgetting /= experiences - 1

    if args.logger:
        print(condensation_args)
        print('final forgetting: ', final_forgetting)
        print('final accuracy: ', final_accuracy)
    return final_forgetting, final_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ex', type=int, default=500,
                        help='number of examples used')
    parser.add_argument('-m', '--memory', type=int, default=100,
                        help='total memory size')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'fashion', 'svhn'])
    parser.add_argument('--model_name', type=str, default='mlp',
                        choices=['mlp', 'resnet', 'mlp_paper', 'cnn'])
    parser.add_argument('--ol', type=int, default=20,
                        help='number of iterations in the outer loop')
    parser.add_argument('--il', type=int, default=3,
                        help='number of iterations in the inner loop')
    parser.add_argument('--lr_w', type=float, default=0.01,
                        help='learning rate to optimize the model used to condense the images')
    parser.add_argument('--run', type=str, default=None,
                        help='Name of the wandb run if not specified wandb will not be used')
    parser.add_argument('--l2_w', type=float, default=0.0,
                        help='l2 weight decay used in the optimiizer of the coefficient of the linear combination')
    parser.add_argument('--plugin', type=str, default='lcgm', choices=['olcgm', 'lcgm', 'rr', 'orr', 'gm', 'ogm'],
                        help='plugin to use: rr is random replay')
    parser.add_argument('--logger', type=int, default=1, choices=[0, 1],
                        help='1 if you want to log the metrics')
    parser.add_argument('--condense_nw', action='store_true',
                        help='1 if you want to condense the new images when added to the memory')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--mb_size', type=int, default=10,
                        help='size of the mini-batchs')
    parser.add_argument('-k', type=int, default=5,
                        help='condensation rate')
    parser.add_argument('--val_size', type=int, default=0,
                        help='validation size')
    parser.add_argument('--seed', '-s', type=int, default=0)

    args = parser.parse_args()

    print(run_experiment(args))
