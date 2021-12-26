"""
@Time : 2021/7/516:06
@Auth : 周俊贤
@File ：run_BERT_avg.py
@DESCRIPTION:

"""

import argparse
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel, BertConfig

from data.dataset import load_STS_data, TestDataset

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--pooling', type=str, default="cls")

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(args.model_path)

dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")


class model(nn.Module):
    def __init__(self, model_path, ):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           output_hidden_states=True)
        if args.pooling == "cls":
            output = output[0][:, -1, :]
        elif args.pooling == "first_last":
            hidden_states = output.hidden_states
            output = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        return output


def test(test_data, model):
    traget_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids, traget_attention_mask, traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred, source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


if __name__ == '__main__':
    model = model(args.model_path).to(device)
    deving_data = TestDataset(dev_data, tokenizer, args.max_len)
    testing_data = TestDataset(test_data, tokenizer, args.max_len)
    corrcoef = test(deving_data, model)
    print("dev corrcoef: {}".format(corrcoef))
    corrcoef = test(testing_data, model)
    print("test corrcoef: {}".format(corrcoef))
