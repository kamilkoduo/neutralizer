from modules.setups.base_neutralizer import *

import pandas as pd

logdir = 'lightning_logs/version_85'

if __name__ == '__main__':

    model = BaseNeutralizer.load_from_checkpoint(f'{logdir}/checkpoints/epoch=163.ckpt')
    model.eval()

    jaffe_dm = JAFFEDataModule(neutral=True,
                               img_size=data.config.IMG_SIZE_DEFAULT)
    jaffe_dm.setup(scenario='exp')

    embedding_vectors = []
    predicted_labels = []
    true_labels = []
    seen = []

    correct = 0
    all = 0
    for sample in jaffe_dm.test_dataloader():
        l, n, exp = model.forward(sample)

        embedding_vectors.append(l.view(-1).detach().numpy())

        predicted_labels.append(Expression(int(exp)).name)
        true_labels.append(Expression(int(sample[1]['exp'])).name)
        seen.append(0)

        if exp == sample[1]['exp']:
            correct += 1

        all += 1

    print(f'val accuracy : {correct / all} from {all}')

    correct = all = 0
    #
    # for sample in jaffe_dm.train_dataloader():
    #     l, n, exp = model.forward(sample)
    #
    #     embedding_vectors.append(l.view(-1).detach().numpy())
    #
    #     predicted_labels.append(Expression(int(exp)).name)
    #     true_labels.append(Expression(int(sample[1]['exp'])).name)
    #     seen.append(1)
    #
    #     if exp == sample[1]['exp']:
    #         correct += 1
    #
    #     all += 1
    # print(f'train accuracy : {correct / all}')

    pd.DataFrame(embedding_vectors).to_csv(f'{logdir}/embeddings/vectors.csv', header=False, index=False, sep='\t')

    d = {'true': true_labels, 'predicted': predicted_labels}
    pd.DataFrame(data=d).to_csv(f'{logdir}/embeddings/labels.csv', header=True, index=False, sep='\t')
