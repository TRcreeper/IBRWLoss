#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
save_steps=50000
q_max_len=32
p_max_len=128
training_mode=oq.nll
train_data_name=dr_oq
train_data_dir=./msmarco_passage/newprocess/train_data/${train_data_name}
model_name=${train_data_name}
model_dir=./msmarco_passage/newmodels/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
save_path=./msmarco_passage/newresults/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
gpu_id=0


query_dir=./output
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 512 \
  --encode_in_path ${query_dir}/train.query.json \
  --encoded_save_path ${save_path}/train.query.pt

index_type=Flat
index_dir=${save_path}/${index_type}

python -m tevatron.faiss_retriever \
  --query_reps ${save_path}/train.query.pt \
  --passage_reps ${save_path}/corpus_128.pt \
  --index_type ${index_type} \
  --batch_size 16 \
  --depth 1000 \
  --save_ranking_file ${index_dir}/train.query.top1k.run.txt \
  --save_index \
  --save_index_dir ${index_dir}
  