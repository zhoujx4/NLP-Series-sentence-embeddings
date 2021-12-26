"""
@Time : 2021/7/516:07
@Auth : 周俊贤
@File ：run_BERT_whitening.py
@DESCRIPTION:

"""
import argparse
import os
import pickle
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from data.dataset import load_STS_data

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--save_path', type=str, default="./output/whitening")
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--pooling', type=str, default="first_last")
parser.add_argument('--dim', type=int, default=768)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")


class model(nn.Module):
    def __init__(self, model_path, ):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = torch.stack([x1[0][:, 0, :], x1[0][:, -1, :]], dim=1)
        output = torch.mean(output, dim=1)
        return output


def sents_to_vecs(sents, tokenizer, model, pooling, max_length):
    vecs = []
    for sent in tqdm(sents, total=len(sents)):
        vec = sent_to_vec(sent, tokenizer, model, pooling, max_length)
        vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def sent_to_vec(sent, tokenizer, model, pooling, max_length):
    with torch.no_grad():
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

        if pooling == 'first_last':
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(args.pooling))

        vec = output_hidden_state.cpu().numpy()[0]
    return vec


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s ** 0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel[:, :args.dim])
    return normalize(vecs)


def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(path, 'wb') as f:
        pickle.dump(whiten, f)
    return path


def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


def test(test_data, model):
    label_list = [int(x[2]) for x in test_data]
    label_list = np.array(label_list)

    sent1_embeddings, sent2_embeddings = [], []
    for sent in tqdm(test_data, total=len(test_data), desc="get sentence embeddings!"):
        vec = sent_to_vec(sent[0], tokenizer, model, args.pooling, args.max_len)
        sent1_embeddings.append(vec)
        vec = sent_to_vec(sent[1], tokenizer, model, args.pooling, args.max_len)
        sent2_embeddings.append(vec)
    target_embeddings = np.vstack(sent1_embeddings)
    target_embeddings = transform_and_normalize(target_embeddings, kernel, bias)  # whitening
    source_embeddings = np.vstack(sent2_embeddings)
    source_embeddings = transform_and_normalize(source_embeddings, kernel, bias)  # whitening

    similarity_list = F.cosine_similarity(torch.Tensor(target_embeddings),
                                          torch.tensor(source_embeddings))
    similarity_list = similarity_list.cpu().numpy()
    corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path).to(device)

    output_filename = "{}-whiten.pkl".format(args.pooling)
    output_path = os.path.join(args.save_path, output_filename)
    if not os.path.exists(output_path):
        train_data = load_STS_data("./data/STS-B/cnsd-sts-train.txt")
        sents = [x[0] for x in train_data] + [x[1] for x in train_data]
        print("Transfer sentences to BERT embedding vectors.")
        vecs_train = sents_to_vecs(sents, tokenizer, model, args.pooling, args.max_len)
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([vecs_train])
        save_whiten(output_path, kernel, bias)
    else:
        kernel, bias = load_whiten(output_path)
    corrcoef = test(dev_data, model)
    print("dev corrcoef: {}".format(corrcoef))
    corrcoef = test(test_data, model)
    print("test corrcoef: {}".format(corrcoef))
