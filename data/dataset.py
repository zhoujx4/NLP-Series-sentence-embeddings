"""
@Time : 2021/7/516:00
@Auth : 周俊贤
@File ：STS-B_dataset.py
@DESCRIPTION:

"""
import json


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list


def load_snli_vocab(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data


def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = int(d[3])
            data.append([sentence1, sentence2, score])
    return data
