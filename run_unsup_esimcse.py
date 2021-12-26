import logging
import math
from datetime import datetime

import torch
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sentence_transformers import models, losses
from sentence_transformers.ConSERT import ConSERT
from sentence_transformers.ESimCSE import ESimCSE
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

from data.dataset import load_STS_data

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# 训练参数
model_name = '/data/junxian/PTMs/chinese-macbert-base'
train_batch_size = 50
num_epochs = 4
model_name = '/data/junxian/PTMs/chinese-roberta-wwm-ext'
train_batch_size = 30
num_epochs = 2
max_seq_length = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = "cuda:0"

# 模型保存路径
model_save_path = '/data/junxian/NLP-Series-sentence-embeddings/output/stsb_esimcse-{}-{}-{}'.format("macbert",
                                                                                                     train_batch_size,
                                                                                                     datetime.now().strftime(
                                                                                                         "%Y-%m-%d_%H-%M-%S"))

# 建立模型
moco_encoder = SentenceTransformer(model_name, device=device).to(device)
moco_encoder.__setattr__("max_seq_length", max_seq_length)
model = ESimCSE(model_name, device=device, dup_rate=0.36, q_size=256)
model.__setattr__("max_seq_length", max_seq_length)

# 准备训练集
sts_vocab = load_STS_data("/data/junxian/STS-B/cnsd-sts-train.txt")
all_vocab = [x[0] for x in sts_vocab] + [x[1] for x in sts_vocab]
model_save_path = '/data/junxian/NLP-Series-sentence-embeddings/output/stsb_simcse-{}-{}-{}'.format("macbert",
                                                                                                    train_batch_size,
                                                                                                    datetime.now().strftime(
                                                                                                        "%Y-%m-%d_%H-%M-%S"))

# 建立模型
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# word_embedding_model.auto_model.attention_probs_dropout_prob = 0.1  #
# word_embedding_model.auto_model.hidden_dropout_prob = 0.1  #
# pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode="cls",
#                                pooling_mode_cls_token=True)
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
model = ESimCSE(model_name, device=device)
model.__setattr__("max_seq_length", max_seq_length)
model.moco_encoder.__setattr__("max_seq_length", max_seq_length)

# 准备训练集
sts_vocab = load_STS_data("/data/junxian/STS-B/cnsd-sts-train.txt")
# all_vocab = [x[0] for x in sts_vocab] + [x[1] for x in sts_vocab]
all_vocab = [x[0] for x in sts_vocab]
>>>>>>> bb0431f... update
simCSE_data = all_vocab
print("The len of SimCSE unsupervised data is {}".format(len(simCSE_data)))
train_samples = []
for data in all_vocab:
    train_samples.append(InputExample(texts=[data, data]))

# 准备验证集和测试集
dev_data = load_STS_data("/data/junxian/STS-B/cnsd-sts-dev.txt")
test_data = load_STS_data("/data/junxian/STS-B/cnsd-sts-test.txt")
dev_samples = []
test_samples = []
for data in dev_data:
    dev_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))
for data in test_data:
    test_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))

# 初始化评估器
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='sts-dev',
                                                                 main_similarity=SimilarityFunction.COSINE)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='sts-test',
                                                                  main_similarity=SimilarityFunction.COSINE)

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss_embeddings(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# 模型训练
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          moco_encoder=moco_encoder,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          show_progress_bar=False,
          output_path=model_save_path,
          optimizer_params={'lr': 2e-5},
          use_amp=False  # Set to True, if your GPU supports FP16 cores
          )

# 测试集上的表现
model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
