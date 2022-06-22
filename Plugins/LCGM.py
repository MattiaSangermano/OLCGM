from avalanche.training.plugins.replay import ReplayPlugin,\
    RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.replay import StoragePolicy
from typing import Dict, Optional
from utils import save_images
from condensations import condenseImagesLinearComb, condenseImagesOriginalGradientMatching

class LCGM(ReplayPlugin):

    def __init__(self, mem_size=200, wandb_logger=None, **kwargs):
        self.ext_mem = {}
        storage_policy = GradientMatchingStoragePolicy(
            ext_mem=self.ext_mem, mem_size=mem_size, adaptive_size=True, 
            selection_strategy=RandomExemplarsSelectionStrategy(),
            wandb_logger=wandb_logger, **kwargs)
        super().__init__(mem_size, storage_policy)

    def before_training_exp(self, strategy, num_workers=0, shuffle=True,
                            **kwargs):
        if strategy.training_exp_counter > 0:
            strategy.dataloader = ReplayDataLoader(
                data=strategy.adapted_dataset,
                memory=AvalancheConcatDataset(self.ext_mem.values()),
                num_workers=num_workers,
                batch_size=strategy.train_mb_size,
                oversample_small_tasks=True,
                shuffle=shuffle, drop_last=True)


class GradientMatchingStoragePolicy(StoragePolicy):
    def __init__(self, ext_mem: Dict, mem_size: int, adaptive_size: bool = True,
                 total_num_classes: int = -1, selection_strategy:
                 Optional["ClassExemplarsSelectionStrategy"] = None,
                 wandb_logger=None, lr_net=0.01,
                 iteration=1, outer_loop=1, lr_w=0.1, l2_w=0.0,
                 image_size=(1, 28, 28), inner_loop=1,condense_new_data=False,
                 dataset='mnist', debug=False, plugin=''):

        super().__init__(ext_mem, mem_size)
        self.selection_strategy = selection_strategy or \
            RandomExemplarsSelectionStrategy()
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.lr_net = lr_net
        self.iteration = iteration
        self.outer_loop = outer_loop
        self.image_size = image_size
        self.lr_w = lr_w
        self.l2_w = l2_w
        self.inner_loop = inner_loop
        self.dataset = dataset
        self.wandb_logger = wandb_logger
        self.condense_new_data = condense_new_data
        self.debug = debug
        self.plugin = plugin
        self.statistics = {}
        if not self.adaptive_size:
            assert self.total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

    def __call__(self, strategy: "BaseStrategy", **kwargs):
        new_data = strategy.experience.dataset
        current_exp = strategy.training_exp_counter
        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # how many classes to divide the memory over
        div_cnt = len(self.seen_classes) if self.adaptive_size \
            else self.total_num_classes
        class_mem_size = self.mem_size // div_cnt
        class_rem_value = self.mem_size % div_cnt

        if strategy.training_exp_counter > 0:
            # class_rem_value = self.mem_size % div_cnt
            images_to_condense = {}
            num_condensed_images = {}
            for c, c_mem in self.ext_mem.items():
                sorted_idxs = self.selection_strategy.make_sorted_indices(
                    strategy, c_mem)
                curr_size = class_mem_size
                if class_rem_value > 0: curr_size += 1
                class_rem_value -= 1

                n = 2 * curr_size - len(c_mem)

                if (curr_size - n) > 0:
                    num_condensed_images[c] = curr_size - n
                    idxs_to_keep = sorted_idxs[:n]
                    idxs_to_condense = sorted_idxs[n:]

                    images_to_condense[c] = AvalancheSubset(
                        c_mem, idxs_to_condense)

                    self.ext_mem[c] = AvalancheSubset(
                                        self.ext_mem[c],
                                        idxs_to_keep)
            if self.plugin == 'lcgm':
                condensed_images = condenseImagesLinearComb(self,
                                    images_to_condense, num_condensed_images, strategy, log='mem')
            elif self.plugin == 'gm':
                condensed_images = condenseImagesOriginalGradientMatching(self,
                                    images_to_condense, num_condensed_images, strategy, log='mem')
            else:
                assert False, 'Wrong condensation type'

            for c, c_condensed in condensed_images.items():
                self.ext_mem[c] = AvalancheConcatDataset([self.ext_mem[c], c_condensed]) 

        num_new_condensed_images = {}
        for c, c_dataset in cl_datasets.items():
            curr_size = class_mem_size
            if class_rem_value > 0: curr_size += 1
            class_rem_value -= 1
            if self.condense_new_data:
                num_new_condensed_images[c] = curr_size
            else:
                c_ds_idxs = self.selection_strategy.make_sorted_indices(
                    strategy, c_dataset)[:curr_size]
                cl_datasets[c] = AvalancheSubset(cl_datasets[c], c_ds_idxs)

        if self.condense_new_data:
            if self.debug:
                print('Condensing new images ...')
            for c, c_dataset in cl_datasets.items():
                c_ds_idxs = self.selection_strategy.make_sorted_indices(
                        strategy, c_dataset)[:5*curr_size] # TODO: occhio 5*
                cl_datasets[c] = AvalancheSubset(cl_datasets[c], c_ds_idxs)

            condensed_new_images = self.condenseImagesLinearComb(
                cl_datasets, num_new_condensed_images, strategy, log='new')

            for c, c_condensed_new in condensed_new_images.items():
                cl_datasets[c] = c_condensed_new

        for c, c_dataset in cl_datasets.items():
            self.ext_mem[c] = c_dataset

        assert len(AvalancheConcatDataset(self.ext_mem.values())) <= self.mem_size, 'Memory exeded max size'

        if self.debug:
            save_images(self.dataset, self.mem_size, AvalancheConcatDataset(self.ext_mem.values())[:][0],
                        str(current_exp) + '_memory', 'grid', self.wandb_logger)
