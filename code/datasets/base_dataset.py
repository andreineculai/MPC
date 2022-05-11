import torch
import torch.utils.data
import torchvision

class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self, config, split='train'):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.test_queries = []
        self.config = config
        self.split = split
    def get_loader(self, pin_memory=True):

        if self.split == 'test' or self.split == 'val':
            num_workers = self.config.dataloader.test_num_workers
            persistent_workers = False
        else:
            num_workers = self.config.dataloader.num_workers
            persistent_workers = True

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.image_to_caption_collate_fn,
            persistent_workers=persistent_workers)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError

    def get_default_transform(self):
        return torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225]),
        ])

    def get_augmentation_transform(self):
        return torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.75, 1.33)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225]),
                torchvision.transforms.RandomErasing(self.config.dataloader.get('random_erasing', 0))
            ])

    def image_to_caption_collate_fn(self, data):
        return data
