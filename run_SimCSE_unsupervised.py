"""
@Time : 2021/7/516:05
@Auth : 周俊贤
@File ：run_CSE_unsupervised.py
@DESCRIPTION:

"""
import argparse
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig

from data.dataset import load_STS_data, TestDataset

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--save_path', type=str, default="./output/SimCSE_supervised.pt")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=4e-5)
parser.add_argument('--max_len', type=int, default=40)
parser.add_argument('--early_stop', type=int, default=10)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(args.model_path)
Config = BertConfig.from_pretrained(args.model_path)
Config.attention_probs_dropout_prob = 0.3
Config.hidden_dropout_prob = 0.3

# snil_vocab = load_snli_vocab("./data/cnsd-snli/cnsd_snli_v1.0.trainproceed.txt")
sts_vocab = load_STS_data("./data/STS-B/cnsd-sts-train.txt")
# all_vocab = [x["origin"] for x in snil_vocab] + [x[0] for x in sts_vocab]
all_vocab = [x[0] for x in sts_vocab] + [x[1] for x in sts_vocab]
# simCSE_data = np.random.choice(all_vocab, 10000)
simCSE_data = all_vocab
print("The len of SimCSE unsupervised data is {}".format(len(simCSE_data)))

dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")


class model(nn.Module):
    def __init__(self, model_path, ):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=Config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = x1.last_hidden_state[:, 0]
        return output


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        sample = self.tokenizer([source, source],
                                max_length=self.max_len,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


def compute_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device='cuda')
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device='cuda') * 1e12
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, testdata, model, optimizer):
    model.train()
    best_corrcoef = 0
    for epoch in range(8):
        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch:{}".format(epoch)):
            input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(device)
            attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            loss = compute_loss(pred)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 20 == 0:
                model.eval()  # 意味着显示关闭丢弃dropout
                corrcoef = test(testdata, model)
                model.train()
                print("\nSpearman's correlation：{:4f}".format(corrcoef))
                if corrcoef > best_corrcoef:
                    best_corrcoef = corrcoef
                    torch.save(model.state_dict(), args.save_path)
                    print("Update the best spearman's correlation：{:4f}，saving model！".format(corrcoef))


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # data
    training_data = TrainDataset(simCSE_data, tokenizer, args.max_len)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, num_workers=16)
    deving_data = TestDataset(dev_data, tokenizer, args.max_len)
    testing_data = TestDataset(test_data, tokenizer, args.max_len)
    # train
    train(train_dataloader, deving_data, model, optimizer)
    # test
    print("train finish!")
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    corrcoef = test(deving_data, model)
    print("dev corrcoef: {}".format(corrcoef))
    corrcoef = test(testing_data, model)
    print("test corrcoef: {}".format(corrcoef))
