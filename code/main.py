import os
import fire
import torch.backends.cudnn as cudnn
from third_party.pcme.config.config import parse_config
from trainers.trainer_factory import trainer_factory
from third_party.pcme.datasets.vocab import Vocabulary
from pathlib import Path
import pprint

pp = pprint.PrettyPrinter(indent=4)

CONFIG_ROOT = 'resources/configs/'

def main(config_name='default.yaml',
         mode='train',
         path=None,
         comment=None,
         **kwargs):

    cudnn.benchmark = False
    if path is not None:
        config = parse_config(os.path.join(Path(path).parent.absolute(), 'config.yaml'),
                              strict_cast=False,
                              **kwargs)
    else:
        config = parse_config(os.path.join(CONFIG_ROOT, config_name),
                              strict_cast=False,
                              **kwargs)
    config.update(mode=mode, path=path, comment=comment)

    if mode == 'train':
        trainer = trainer_factory(config.trainer, config, path)
        trainer.train()
    elif mode == 'eval':
        trainer = trainer_factory(config.trainer, config, path)
        out = trainer.eval(trainer.test_dataset)
        pp.pprint(out)
        with open(os.path.join(Path(path).parent.absolute(), 'results.txt'),
                  'wt') as fout:
            pprint.pprint(out, stream=fout)

if __name__ == '__main__':
    fire.Fire(main)
