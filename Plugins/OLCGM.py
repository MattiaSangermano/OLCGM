import random 

from avalanche.training.plugins.replay import ReplayPlugin,\
    RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.replay import StoragePolicy

from typing import Dict, Optional
from utils import save_images

from condensations import condenseImagesLinearComb, condenseImagesOriginalGradientMatching


class OLCGM(ReplayPlugin):

    def __init__(self, mem_size=200, wandb_logger=None, **kwargs):
        self.ext_mem = {}
        storage_policy = OnlineGradientMatchingStoragePolicy(
            ext_mem=self.ext_mem, mem_size=mem_size, adaptive_size=True, 
            selection_strategy=RandomExemplarsSelectionStrategy(),
            wandb_logger=wandb_logger, **kwargs)
        super().__init__(mem_size, storage_policy)

    def before_training_exp(self, strategy, num_workers=0, shuffle=True,
                            **kwargs):

        if strategy.training_exp_counter > 0:
            memory = AvalancheConcatDataset(self.ext_mem.values())
            indices = list(range(len(memory)))
            random.shuffle(indices)
            indices = indices[:strategy.train_mb_size]
            mini_batch_memory = AvalancheSubset(memory, indices)

            strategy.dataloader = ReplayDataLoader(
                data=strategy.adapted_dataset,
                memory=mini_batch_memory,
                num_workers=num_workers,
                batch_size= 2 * strategy.train_mb_size,
                oversample_small_tasks=False,
                shuffle=shuffle, drop_last=True)

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy(strategy, **kwargs)


class OnlineGradientMatchingStoragePolicy(StoragePolicy):
    def __init__(self, ext_mem: Dict, mem_size: int,
                 adaptive_size: bool = True, total_num_classes: int = -1,
                 lr_net=0.01, wandb_logger=None, selection_strategy:
                 Optional["ClassExemplarsSelectionStrategy"] = None,
                 iteration=1, outer_loop=1, lr_w=0.1, l2_w=0.0,
                 image_size=(1, 28, 28), inner_loop=1, dataset='mnist',
                 debug=False, k=5, dl=1, plugin='', condense_new_data=False):

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
        self.k = k
        self.counter_k = k
        self.dl = dl
        self.plugin = plugin
        self.statistics = {}
        if not self.adaptive_size:
            assert self.total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

    def __call__(self, strategy: "BaseStrategy", **kwargs):

        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        self.counter_k -= 1
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, c_idxs)

        # Update seen classes
        previous_len = len(self.seen_classes)
        self.seen_classes.update(cl_datasets.keys())
        new_classes = previous_len != len(self.seen_classes)

        # How many classes to divide the memory over
        div_cnt = len(self.seen_classes) if self.adaptive_size \
            else self.total_num_classes
        class_mem_size = self.mem_size // div_cnt
        class_rem_value = self.mem_size % div_cnt

        curr_mem_size = len(AvalancheConcatDataset(self.ext_mem.values()))

        if curr_mem_size < self.mem_size:
            # Concat mini-batch to the memory
            class_rem_value = self.fill_memory_with_mini_batch(cl_datasets, class_mem_size, class_rem_value, strategy)

        elif self.counter_k <= 0 and not new_classes:
            self.change_memory_images(cl_datasets, strategy)

        if new_classes:
            self.free_space_for_new_classes(strategy, class_mem_size, class_rem_value,cl_datasets)
        assert len(AvalancheConcatDataset(self.ext_mem.values())) <= self.mem_size, 'Memory exeded max size'

        if self.debug:
            current_exp = strategy.training_exp_counter
            if (current_exp + 5) % 100 == 0:
                save_images(self.dataset, self.mem_size, AvalancheConcatDataset(self.ext_mem.values())[:][0],
                            str(current_exp) + '_memory', 'grid', self.wandb_logger)

    def fill_memory_with_mini_batch(self, cl_datasets, class_mem_size, class_rem_value, strategy): 
        all_keys = list(self.ext_mem.keys())
        for k in cl_datasets.keys():
            if k not in all_keys:
                all_keys.append(k)

        for c in all_keys:
            curr_size = class_mem_size
            if class_rem_value > 0: curr_size += 1
            class_rem_value -= 1
            if c in cl_datasets:
                if c in self.ext_mem and len(cl_datasets[c]) + len(self.ext_mem[c]) > curr_size:
                    idxs = range(0, curr_size - len(self.ext_mem[c]))
                    cl_datasets[c] = AvalancheSubset(cl_datasets[c], indices=idxs)
                if c in self.ext_mem:
                    self.ext_mem[c] = AvalancheConcatDataset([self.ext_mem[c], cl_datasets[c]])
                else:
                    ids = self.selection_strategy.make_sorted_indices(
                        strategy, cl_datasets[c])[:curr_size]
                    self.ext_mem[c] = AvalancheSubset(cl_datasets[c], indices=ids)
        return class_rem_value

    def change_memory_images(self, cl_datasets, strategy):
        # Add new images in memory, freeing up space through 
        # the use of condensation

        self.counter_k = self.k
        images_to_condense = {}
        num_condensed_images = {}
        for c, c_mem in self.ext_mem.items():
            if c in cl_datasets:
                num_ex = min(len(c_mem), len(cl_datasets[c]))
                sorted_idxs_mem = self.selection_strategy.make_sorted_indices(
                        strategy, c_mem)
                sorted_idxs_mb = self.selection_strategy.make_sorted_indices(
                        strategy, cl_datasets[c])

                ids_to_keep_mem = sorted_idxs_mem[num_ex:]
                ids_to_condense_mem = sorted_idxs_mem[:num_ex]
                ids_to_condense_mb = sorted_idxs_mb[:num_ex]

                mem_cond = AvalancheSubset(c_mem, ids_to_condense_mem)
                mb_cond = AvalancheSubset(cl_datasets[c], ids_to_condense_mb)

                num_condensed_images[c] = num_ex
                images_to_condense[c] = AvalancheConcatDataset([mem_cond, mb_cond])

                self.ext_mem[c] = AvalancheSubset(
                                    self.ext_mem[c],
                                    ids_to_keep_mem)

        if self.plugin == 'olcgm':
            condensed_images = condenseImagesLinearComb(
                self, images_to_condense, num_condensed_images,
                strategy, log='mem')
        elif self.plugin == 'ogm':
            condensed_images = condenseImagesOriginalGradientMatching(
                self, images_to_condense, num_condensed_images,
                strategy, log='mem')
        else:
            assert False, 'Wrong condensation type'

        for c, c_condensed in condensed_images.items():
            self.ext_mem[c] = AvalancheConcatDataset([self.ext_mem[c], c_condensed])

    def free_space_for_new_classes(self, strategy, class_mem_size, class_rem_value, cl_datasets):
        # Free space for the new classes by condensing the memory
        self.counter_k = self.k
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

        if len(num_condensed_images) > 0:
            if self.plugin == 'olcgm':
                condensed_images = condenseImagesLinearComb(self,
                    images_to_condense, num_condensed_images, strategy, log='mem')
            elif self.plugin == 'ogm':
                condensed_images = condenseImagesOriginalGradientMatching(self,
                    images_to_condense, num_condensed_images, strategy, log='mem')
            else:
                assert False, 'Wrong condensation type'

            for c, c_condensed in condensed_images.items():
                self.ext_mem[c] = AvalancheConcatDataset([self.ext_mem[c], c_condensed]) 

            # Insertion of new classes in memory
            for c, c_dataset in cl_datasets.items():
                if c not in self.ext_mem:
                    curr_size = class_mem_size
                    if class_rem_value > 0: curr_size += 1
                    class_rem_value -= 1
                    c_ds_idxs = self.selection_strategy.make_sorted_indices(
                        strategy, c_dataset)[:curr_size]
                    cl_datasets[c] = AvalancheSubset(cl_datasets[c], c_ds_idxs)

            for c, c_dataset in cl_datasets.items():
                if c not in self.ext_mem:
                    self.ext_mem[c] = c_dataset

    