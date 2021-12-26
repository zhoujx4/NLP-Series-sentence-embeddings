import json
import logging
import math
import os
import queue
import random
import shutil
import stat
import tempfile
from collections import OrderedDict
from distutils.dir_util import copy_tree
from typing import Dict, Tuple, Iterable, Type, Callable
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import HfApi, HfFolder, Repository
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

from . import __MODEL_HUB_ORGANIZATION__
from . import __version__
from .evaluation import SentenceEvaluator
from .model_card_templates import ModelCardTemplate
from .models import Transformer, Pooling
from .util import import_from_string, batch_to_device, fullname, snapshot_download

logger = logging.getLogger(__name__)


class ESimCSE(SentenceTransformer):
    def __init__(self, model_name_or_path=None, modules=None, device=None, cache_folder=None, q_size=256,
                 dup_rate=0.32):
        SentenceTransformer.__init__(self, model_name_or_path, modules, device, cache_folder)
        # self.moco_encoder = SentenceTransformer(model_name_or_path, device=device)  #
        # moco_encoder = moco_encoder  #
        self.gamma = 0.99
        self.q = []
        self.q_size = q_size
        self.dup_rate = dup_rate

    def word_repetition(self, sentence_feature):
        input_ids, attention_mask, token_type_ids = sentence_feature['input_ids'].cpu().tolist(
        ), sentence_feature['attention_mask'].cpu().tolist(), sentence_feature['token_type_ids'].cpu().tolist()
        bsz, seq_len = len(input_ids), len(input_ids[0])
        # print(bsz,seq_len)
        repetitied_input_ids = []
        repetitied_attention_mask = []
        repetitied_token_type_ids = []
        rep_seq_len = seq_len
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            actual_len = sum(sample_mask)

            cur_input_id = input_ids[bsz_id]
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=dup_len)

            r_input_id = []
            r_attention_mask = []
            r_token_type_ids = []
            for index, word_id in enumerate(cur_input_id):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])
                    r_token_type_ids.append(token_type_ids[bsz_id][index])

                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])
                r_token_type_ids.append(token_type_ids[bsz_id][index])

            after_dup_len = len(r_input_id)
            # assert after_dup_len==actual_len+dup_len
            repetitied_input_ids.append(r_input_id)  # +rest_input_ids)
            repetitied_attention_mask.append(
                r_attention_mask)  # +rest_attention_mask)
            repetitied_token_type_ids.append(
                r_token_type_ids)  # +rest_token_type_ids)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(bsz):
            after_dup_len = len(repetitied_input_ids[i])
            pad_len = rep_seq_len - after_dup_len
            repetitied_input_ids[i] += [0] * pad_len
            repetitied_attention_mask[i] += [0] * pad_len
            repetitied_token_type_ids[i] += [0] * pad_len

        repetitied_input_ids = torch.LongTensor(repetitied_input_ids).to(self.device)
        repetitied_attention_mask = torch.LongTensor(repetitied_attention_mask).to(self.device)
        repetitied_token_type_ids = torch.LongTensor(repetitied_token_type_ids).to(self.device)
        return {"input_ids": repetitied_input_ids, 'attention_mask': repetitied_attention_mask,
                'token_type_ids': repetitied_token_type_ids}

    # def save(self, path, model_name=None, create_model_card=True):
    # """
    # Saves all elements for this seq. sentence embedder into different sub-folders
    # :param path: Path on disc
    # :param model_name: Optional model name
    # :param create_model_card: If True, create a README.md with basic information about this model
    # """
    # if path is None:
    #     return
    #
    # os.makedirs(path, exist_ok=True)
    #
    # logger.info("Save model to {}".format(path))
    # modules_config = []
    #
    # # Save some model info
    # if '__version__' not in self._model_config:
    #     self._model_config['__version__'] = {
    #         'sentence_transformers': __version__,
    #         'transformers': transformers.__version__,
    #         'pytorch': torch.__version__,
    #     }
    #
    # with open(os.path.join(path, 'config_sentence_transformers.json'), 'w') as fOut:
    #     json.dump(self._model_config, fOut, indent=2)
    #
    # # Save modules
    # for idx, name in enumerate(self._modules):
    #     module = self._modules[name]
    #     if idx == 0 and isinstance(module, Transformer):  # Save transformer model in the main folder
    #         model_path = path + "/"
    #     elif isinstance(module, SentenceTransformer):
    #         continue
    #     else:
    #         model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)
    #
    #     os.makedirs(model_path, exist_ok=True)
    #     module.save(model_path)
    #     modules_config.append(
    #         {'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})
    #
    # with open(os.path.join(path, 'modules.json'), 'w') as fOut:
    #     json.dump(modules_config, fOut, indent=2)
    #
    # # Create model card
    # if create_model_card:
    #     self._create_model_card(path, model_name)

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            moco_encoder: SentenceTransformer = None,  #
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
             "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),
             "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
             "max_grad_norm": max_grad_norm}, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}",
                                                                                                     info_loss_functions).replace(
            "{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data

                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    features[1] = self.word_repetition(sentence_feature=features[1])
                    batch_size = labels.size(0)

                    negative_samples = None
                    if len(self.q) > 0:
                        negative_samples = torch.vstack(
                            self.q[:self.q_size])  # (q_size,768)
                    if len(self.q) + batch_size >= self.q_size:
                        del self.q[:batch_size]

                    with torch.no_grad():
                        moco_encoder[0].auto_model.encoder.config.attention_probs_dropout_prob = 0.0
                        moco_encoder[0].auto_model.encoder.config.hidden_dropout_prob = 0.0
                        self.q.extend(moco_encoder(
                            features[0])['sentence_embedding'])

                    for encoder_param, moco_encoder_param in zip(self.parameters(),
                                                                 moco_encoder.parameters()):
                        moco_encoder_param.data = self.gamma * \
                                                  moco_encoder_param.data + (1. - self.gamma) * encoder_param.data
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels, negative_samples)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels, negative_samples)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


if __name__ == '__main__':
    model = ESimCSE()