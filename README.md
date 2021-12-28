# 项目说明
本项目是句子embedding、语义匹配任务的代码复现,是在sbert库的基础上做的二次开发  
包括以下方法：
- BERT_avg
- BERT_whitening
- SBERT
- SimCSE
- ConSERT
- ESimCSE  

部分实验效果如下：   
 
|                                | Chines dev | Chines test |   
| ------------------------------ | ----- | --- |    
| BERT-avg(CLS-token)      | 54.39 | 48.34 |     
| BERT-whitening    | 74.70 | 66.98 |     
| SBERT（SNLI）             | 74.40 | 72.41 |       
| SBERT（STS-B）        | 82.10 | 77.69 |     
| SimCSE        | 76.55 | 71.91 |     
| ConSERT        | 78.35 | 72.48 |    
| ESimCSE        | 78.82 | 71.81 |    

>更具体方法原理和效果，请看我的知乎博客https://zhuanlan.zhihu.com/p/450993543?
# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0

## 数据集
https://github.com/pluto-junzeng/CNSD 中的Chinese-SNLI数据集和Chinese-STS-B数据集 

