import logging
import math
from datetime import datetime

import numpy as np
import torch
from data.dataset import load_STS_data, load_snli_vocab
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# 训练参数
model_name = '/data/junxian/PTMs/chinese-macbert-base'
train_batch_size = 64
num_epochs = 5
max_seq_length = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = "cuda:1"

# 模型保存路径
model_save_path = '/data/junxian/NLP-Series-sentence-embeddings/output/stsb_simcse-{}-{}-{}'.format("macbert",
                                                                                                    train_batch_size,
                                                                                                    datetime.now().strftime(
                                                                                                        "%Y-%m-%d_%H-%M-%S"))

# 建立模型
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode="cls",
                               pooling_mode_cls_token=True)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# 准备训练集
snil_vocab = load_snli_vocab("/data/junxian/cnsd-snli/cnsd_snli_v1.0.trainproceed.txt")
np.random.shuffle(snil_vocab)
print("The len of snil supervised data is {}".format(len(snil_vocab)))
train_samples = []
for data in snil_vocab:
    train_samples.append(InputExample(texts=[data["origin"], data["entailment"], data["contradiction"]]))

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
train_loss = losses.TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.5)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# 模型训练
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
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
