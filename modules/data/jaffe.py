from pathlib import Path

from torchvision import transforms

from modules.data import config, common, transforms as c_transforms
from modules.data.common import Expression, CommonDataset, CommonDataModule, ExpressionBatchSampler, \
    SingleExpressionBatchSampler

ExpEncJAFFE = {
    'NE': Expression.NEUTRAL,
    'AN': Expression.ANGRY,
    'DI': Expression.DISGUST,
    'FE': Expression.FEAR,
    'HA': Expression.HAPPY,
    'SA': Expression.SAD,
    'SU': Expression.SURPRISE,
}


def parse_entry_jaffe(entry):
    tokens = entry.name.split('.')
    if tokens[-1] != 'tiff':
        return None

    exp = ExpEncJAFFE[tokens[1][:2]]
    iden = tokens[0]

    return {
        'path': str(entry),
        'exp': exp.value,
        'iden': iden,
    }


def parse_entries_jaffe(data_dir):
    return [parse_entry_jaffe(entry) for entry in Path(data_dir).iterdir()]


class JAFFEDataset(CommonDataset):
    def __init__(self, data_dir=config.RAW_JAFFE_DATA_DIR, neutral=False, img_size=config.IMG_SIZE_DEFAULT):
        super().__init__(
            descriptions=parse_entries_jaffe(data_dir=data_dir),
            neutral=neutral,
            img_size=img_size
        )


class JAFFEDataModule(CommonDataModule):
    def __init__(self, batch_aligned=False, batch_size=32, img_size=config.IMG_SIZE_DEFAULT, num_workers=16):

        super().__init__(dataset_class=JAFFEDataset, batch_aligned=batch_aligned, batch_size=batch_size,
                         img_size=img_size, num_workers=num_workers)

    def prepare_data(self):
        if not Path(config.RAW_JAFFE_DATA_DIR).exists():
            print('JAFFE data not found')
        else:
            print('JAFFE data found')


if __name__ == '__main__':
    # test
    jaffe = JAFFEDataset(neutral=True, img_size=(256, 256))

    sebs = SingleExpressionBatchSampler(jaffe, Expression.NEUTRAL, batch_size=10)
    print(len(sebs))
    print(sebs.batch_size)
    print(sebs.indices[0:0 + sebs.batch_size])

    for s in iter(sebs):
        print(s)
        print([jaffe[i]['desc']['exp'] for i in s])

    # smp = ExpressionBatchSampler(batch_size=10, dataset=jaffe)
    # for s in smp:
    #     print(s)

    # print(smp)
    #
    #
    # print(jaffe.descriptions)
    #
    # img = jaffe[0]['image']
    #
    # print(img.size())
    # print(img)
    #
    # img = c_transforms.default_denormalize(img)
    # img = transforms.ToPILImage()(img)
    # img.show()
    #
    # # one, two = jaffe.exp_split()
    # # print(len(one), len(two), len(jaffe))
    #
    # dm = JAFFEDataModule()
    # dm.prepare_data()
    #
    # dm.setup(neutral=True, expression=Expression.ANGRY, scenario='train', aligned=True)
    # print(len(dm.train_dataloader()))
    #
    # for batch in dm.train_dataloader():
    #     print(batch)
    #
    # dm.setup(neutral=True, expression=Expression.ANGRY, scenario='test')
    # print(len(dm.test_dataloader()))
