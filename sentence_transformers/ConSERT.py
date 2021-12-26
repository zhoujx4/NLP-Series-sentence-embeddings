import random

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device


class ConSERT(SentenceTransformer):
    def __init__(self, model_name_or_path=None, modules=None, device=None, cache_folder=None, cutoff_rate=0.15,
                 close_dropout=False):
        SentenceTransformer.__init__(self, model_name_or_path, modules, device, cache_folder)
        self.cutoff_rate = cutoff_rate
        self.close_dropout = close_dropout

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        sentence_features[1] = self.shuffle_and_cutoff(sentence_features[1])

        return sentence_features, labels

    def shuffle_and_cutoff(self, sentence_feature):
        input_ids, attention_mask = sentence_feature['input_ids'], sentence_feature['attention_mask']
        bsz, seq_len = input_ids.shape
        shuffled_input_ids = []
        cutoff_attention_mask = []
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            num_tokens = sample_mask.sum().int().item()
            cur_input_ids = input_ids[bsz_id]
            if 102 not in cur_input_ids:  # tip:
                indexes = list(range(num_tokens))[1:]
                random.shuffle(indexes)
                indexes = [0] + indexes  # 保证第一个位置是0
            else:
                indexes = list(range(num_tokens))[1:-1]
                random.shuffle(indexes)
                indexes = [0] + indexes + [num_tokens - 1]  # 保证第一个位置是0，最后一个位置是SEP不变
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_input_id = input_ids[bsz_id][total_indexes]
            # print(shuffled_input_id,indexes)
            if self.cutoff_rate > 0.0:
                sample_len = max(int(num_tokens * (1 - self.cutoff_rate)),
                                 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                start_id = np.random.randint(1,
                                             high=num_tokens - sample_len + 1)  # start_id random select from (0,6)，避免删除CLS
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的
                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换
                cutoff_mask = torch.ByteTensor(cutoff_mask).bool().to(input_ids.device)
                shuffled_input_id = shuffled_input_id.masked_fill(cutoff_mask, value=0).to(input_ids.device)
                sample_mask = sample_mask.masked_fill(cutoff_mask, value=0).to(input_ids.device)

            shuffled_input_ids.append(shuffled_input_id)
            cutoff_attention_mask.append(sample_mask)
        shuffled_input_ids = torch.vstack(shuffled_input_ids)
        cutoff_attention_mask = torch.vstack(cutoff_attention_mask)
        return {"input_ids": shuffled_input_ids, "attention_mask": cutoff_attention_mask, "token_type_ids": sentence_feature["token_type_ids"]}


if __name__ == '__main__':
    model = ConSERT()
