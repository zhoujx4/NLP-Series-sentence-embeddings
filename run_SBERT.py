"""
@Time : 2021/7/516:04
@Auth : 周俊贤
@File ：run_SBERT.py
@DESCRIPTION:

"""

# coding=UTF-8


import argparse
import numpy as np
import scipy.stats
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models, evaluation
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from data.dataset import load_STS_data, load_snli_vocab

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default="64")
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--dataset', type=str, default="STS-B")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(args.model_path)
Config = BertConfig.from_pretrained(args.model_path)

snil_vocab = load_snli_vocab("./data/cnsd-snli/cnsd_snli_v1.0.trainproceed.txt")
np.random.shuffle(snil_vocab)
train_data = load_STS_data("./data/STS-B/cnsd-sts-train.txt")
dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")


def test(test_data, model):
    sentences1 = [x[0] for x in test_data]
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    sentences2 = [x[1] for x in test_data]
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    label_list = [x[2] for x in test_data]
    similarity_list = F.cosine_similarity(embeddings1, embeddings2)
    similarity_list = similarity_list.cpu().numpy()
    label_list = np.array(label_list)
    corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation

    return corrcoef


if __name__ == '__main__':
    word_embedding_model = models.Transformer(args.model_path,
                                              max_seq_length=args.max_len)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if args.dataset == "STS-B":
        train_examples = []
        for data in train_data:
            train_examples.append(InputExample(texts=[data[0], data[1]], label=data[2]/5))
        sentences1, sentences2, scores = [], [], []
        for data in dev_data:
            sentences1.append(data[0])
            sentences2.append(data[1])
            scores.append(data[2]/5)
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, batch_size =100, write_csv=True)
        #
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.CosineSimilarityLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=8,
                  warmup_steps=20,
                  evaluator=evaluator,
                  evaluation_steps=40,
                  output_path="./output/SBERT/STS-B")
        corrcoef = test(test_data, model)
        print("test corrcoef: {}".format(corrcoef))
    elif args.dataset == "SNLI":
        train_examples = []
        for data in snil_vocab:
            train_examples.append(InputExample(texts=[data["origin"], data["entailment"], data["contradiction"]]))
        sentences1, sentences2, scores = [], [], []
        for data in dev_data:
            sentences1.append(data[0])
            sentences2.append(data[1])
            scores.append(data[2] / 5)
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, batch_size=100, write_csv=True)
        #
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.TripletLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=2,
                  warmup_steps=200,
                  evaluator=evaluator,
                  evaluation_steps=500,
                  output_path="./output/SBERT/SNLI")
        corrcoef = test(test_data, model)
        print("test corrcoef: {}".format(corrcoef))
