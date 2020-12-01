from pathlib import Path

from networkx.drawing.tests.test_pylab import plt

from modules.data import config
from modules.data.common import Expression, CommonDataset, CommonDataModule
from torchvision import transforms

from modules.data import transforms as c_transforms

ExpEncTFEID = {
    'ng': Expression.NEUTRAL,
    'ag': Expression.ANGRY,
    'dg': Expression.DISGUST,
    'fg': Expression.FEAR,
    'hg': Expression.HAPPY,
    'sg': Expression.SAD,
    'pg': Expression.SURPRISE,
}


def parse_entry_tfeid(entry):
    tokens = entry.name.split('.')
    if tokens[-1] != 'jpg':
        return None

    tokens = tokens[0].split('_')

    exp = ExpEncTFEID[tokens[-1][:2]]
    iden = tokens[0]

    return {
        'path': str(entry),
        'exp': exp.value,
        'iden': iden,
    }


def parse_entries_tfeid(data_dir):
    subdirs = [c for c in Path(data_dir).iterdir() if c.is_dir()]

    return [parse_entry_tfeid(entry) for subdir in subdirs for entry in subdir.iterdir()]

class TFEIDDataset(CommonDataset):
    def __init__(self, data_dir=config.RAW_TFEID_DATA_DIR, neutral=False, img_size=config.IMG_SIZE_DEFAULT):

        super().__init__(
            descriptions=parse_entries_tfeid(data_dir=data_dir),
            neutral=neutral,
            img_size=img_size
        )


class JAFFEDataModule(CommonDataModule):
    def __init__(self,  batch_size=32, img_size=config.IMG_SIZE_DEFAULT, num_workers=16):

        super().__init__(dataset_class=TFEIDDataset, batch_size=batch_size, img_size=img_size, num_workers=num_workers)

    def prepare_data(self):
        nt = 'not' if not Path(config.RAW_TFEID_DATA_DIR).exists() else ''
        print(f'TFEID data {nt} found')


if __name__ == '__main__':
    # test
    tfeid = TFEIDDataset(neutral=True, img_size=(256,256))

    print(tfeid.descriptions)

    img = tfeid[0]['image']

    print(img.size())
    print(img)

    img = c_transforms.default_denormalize(img)
    img = transforms.ToPILImage()(img)
    img.show()

    # one, two = jaffe.exp_split()
    # print(len(one), len(two), len(jaffe))
#
#     dm = JAFFEDataModule()
#     dm.prepare_data()
#
#     dm.setup(neutral=True, expression=Expression.ANGRY, scenario='train')
#     print(len(dm.train_dataloader()))
#
#     dm.setup(neutral=True, expression=Expression.ANGRY, scenario='test')
#     print(len(dm.test_dataloader()))
#
#
#
# if __name__ == '__main__':
#     entries = parse_entries_tfeid(config.RAW_TFEID_DATA_DIR)
#     print(entries)