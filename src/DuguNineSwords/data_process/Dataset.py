import torch
from torch.utils import data


class Word2vecIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __iter__(self):
        from itertools import chain, cycle
        # return chain.from_iterable(map(self._process_file, cycle(self.file_names)))
        aa = chain.from_iterable(self.inputs)
        return aa


class Word2vecDataset(data.Dataset):
    def __init__(self, parser):
        self.parser = parser

    def __len__(self):
        return self.parser.sentence_num()

    def __getitem__(self, idx):
        pass
        # words = self.parser[idx]
        # two_word = random.sample(words, k=2)
        # indices = self.parser.vocab.lookup_all(two_word)
        # center = indices[0]
        # context = indices[1]
        # neg_samples = self.parser._neg_samples(center, context, k=10)
        # return np.array([center]), np.array([context]), np.array(neg_samples)


if __name__ == '__main__':
    # train_files = [
    #     "part-00000-b9c23c08-91aa-4f48-ade6-dd25c443fbc8-c000.csv",
    #     "part-00001-b9c23c08-91aa-4f48-ade6-dd25c443fbc8-c000.csv",
    #     "part-00002-b9c23c08-91aa-4f48-ade6-dd25c443fbc8-c000.csv",
    #     "part-00003-b9c23c08-91aa-4f48-ade6-dd25c443fbc8-c000.csv"
    # ]
    # train_files = [os.path.join(config.root_dirname, "data/embeddings", file_name) for file_name in train_files]
    #
    # parser = Item2vecParser(train_files)
    from embeddings.kafka_parser import KafkaParser
    parser = KafkaParser()
    dataset = Word2vecIterableDataset(parser)
    num = 0
    for i in dataset:
        print(num, i)
        num += 1
        break
    # print(len(dataset))
    # print(dataset[42])
    training_generator = data.DataLoader(dataset, batch_size=4)

    for i in training_generator:
        print(i)
        break


