import json
import os
import copy
from dataclasses import dataclass
import numpy as np
import random

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    qv_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    # scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        # loss print
        self.total_loss = 0.
        self.total_qvp_nll_loss = 0.
        self.total_qp_lra_loss = 0.
        self.total_pq_lra_loss = 0.
        self.ranking_buckets = [(i, i) for i in range(8)] + [(8, float('inf'))]
        self.bucket_metrics = {bucket: {'count': 0} for bucket in self.ranking_buckets}

        self.record_interval = 100
        self.global_step = 0
        self.record_file_path = 'training_records.csv'
        self.rank_loss_record_file_path = 'rank_loss_contribution.csv'

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()


    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            query_variation: Dict[str, Tensor] = None,
            original_reps: Tensor = None,
            current_step: int = None
    ):
        ## The default ordering of encoding forward: original query -> passage -> query variation
        ## It may affect the DR training slightly.
        if self.train_args.training_mode == 'oq.nll':
            q_hidden, q_reps = self.encode_query(query)
            p_hidden, p_reps = self.encode_passage(passage)
            qv_hidden, qv_reps = None, None
        else:
            raise NotImplementedError('Please choose the correct training mode.')

        if (q_reps is None and qv_reps is None) or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                qv_reps=qv_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.train_args.negatives_x_device:
                if q_reps is not None:
                    q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)
                if qv_reps is not None:
                    qv_reps = self.dist_gather_tensor(qv_reps)
                if original_reps is not None:
                    original_reps.q_reps = self.dist_gather_tensor(original_reps.q_reps)
                    original_reps.p_reps = self.dist_gather_tensor(original_reps.p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            if self.train_args.training_mode == 'oq.nll':
                scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
                scores = scores.view(effective_bsz, -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * self.data_args.train_n_passages

                rankings = scores.argsort(dim=1, descending=True)
                positive_rankings = torch.zeros(scores.size(0), device=scores.device)
                for i in range(scores.size(0)):
                    positive_idx = target[i]
                    ranking_position = (rankings[i] == positive_idx).nonzero(as_tuple=True)[0].item()  # 正样本的排名
                    positive_rankings[i] = ranking_position

                import torch.nn.functional as F
                raw_loss = F.cross_entropy(scores, target, reduction='none')


                alpha = 2.6  
                optimal_rank = 1.0  
                sigma = 1.8  


                extra_weight = torch.exp(- ((positive_rankings - optimal_rank) ** 2) / (2 * sigma ** 2))
                weights = 1.0 + alpha * extra_weight

                weighted_scores = scores * weights.unsqueeze(1)

                for i in range(scores.size(0)):
                    ranking = positive_rankings[i].item()
                    for bucket in self.ranking_buckets:
                        if bucket[0] <= ranking <= bucket[1]:
                            self.bucket_metrics[bucket]['count'] += 1
                            break
                
                rank_loss_contribution = []
                for i in range(scores.size(0)):
                    ranking = int(positive_rankings[i].item())
                    loss_contribution = raw_loss[i].item()
                    rank_loss_contribution.append((ranking, loss_contribution))

                loss = (raw_loss * weights).mean()
                # loss=raw_loss.mean()

                self.global_step += 1

                if self.global_step % self.record_interval == 0:
                    with open(self.record_file_path, 'a') as f:
                        if self.global_step == self.record_interval:
                            f.write('Step,Bucket Start,Bucket End,Query Count\n')
                        for bucket, metrics in self.bucket_metrics.items():
                            bucket_start = bucket[0]
                            bucket_end = bucket[1] if bucket[1] != float('inf') else '8+'
                            count = metrics['count']
                            f.write(f'{self.global_step},{bucket_start},{bucket_end},{count}\n')
                    with open(self.rank_loss_record_file_path, 'a') as f:
                        if self.global_step == self.record_interval:
                            f.write('Step,Positive Ranking,Loss Contribution\n')
                        for ranking, contribution in rank_loss_contribution:
                            f.write(f'{self.global_step},{ranking},{contribution}\n')
                    for bucket in self.bucket_metrics:
                        self.bucket_metrics[bucket]['count'] = 0
                    rank_loss_contribution = []

            else:
                raise NotImplementedError

            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction

            return DenseOutput(
                loss=loss,
                q_reps=q_reps,
                qv_reps=qv_reps,
                p_reps=p_reps
            )

        else:
            return DenseOutput(
                q_reps=q_reps,
                qv_reps=qv_reps,
                p_reps=p_reps
            )

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_args.model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DenseModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DenseModelForInference, self).encode_query(qry)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler
        )
        return model