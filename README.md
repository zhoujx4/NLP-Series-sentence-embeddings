# 项目说明
本项目是句子embedding、语义匹配任务的代码复现
包括以下四种方法
- BERT_avg
- BERT_whitening
- SBERT
- SimCSE  

>具体方法原理和效果，请看我的知乎博客https://zhuanlan.zhihu.com/p/388680608
# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0
# 运行示例
BERT_avg
```
python3 run_BERT_avg.py
--model_path=预训练模型路径
--pooling=first-last
```
BERT_whitening
```
python3 run_BERT_whitening.py
--model_path=预训练模型路径
--pooling=first_last
--dim=768
```
SBERT
```
python3 run_SBERT.py
--model_path=预训练模型路径
--dataset=STS-B
--max_len=80
```
SimCSE(unsupervised)
```
python3 run_SimCSE_unsupervised.py
--model_path=预训练模型路径

```
SimCSE(supervised)
```
python3 run_SimCSE_supervised.py
--model_path=预训练模型路径
```