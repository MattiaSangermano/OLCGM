import random 
from avalanche.training.plugins.replay import ReplayPlugin
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.replay import StoragePolicy
from typing import Optional


class OnlineReplay(ReplayPlugin):

    def __init__(self, mem_size=200, storage_policy: Optional["StoragePolicy"] = None):
        self.ext_mem = {}
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