from torch.utils.data.dataloader import default_collate

class SegCollate(object):
    def __init__(self, batch_aug_fn=None):
        self.batch_aug_fn = batch_aug_fn

    def __call__(self, batch):
        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)
        return default_collate(batch)