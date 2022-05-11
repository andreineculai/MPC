from datasets.coco_dataset import Coco


def dataset_factory(config, split, **kwargs):
    if config.dataset_name == 'coco':
        return Coco(config, split, kwargs.get('clothing_types', None))

