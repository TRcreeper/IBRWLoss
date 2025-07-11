#!/bin/bash

pd_tbs=128
lr=5e-6
epoch=4
save_steps=50000
q_max_len=32
p_max_len=128
training_mode=oq.nll
train_data_name=dr_rand
train_data_dir=./msmarco_passage/newprocess/train_data/${train_data_name}
output_dir=./msmarco_passage/newmodels/${train_data_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}


python build_train.py \
  --tokenizer_name ./model/bert-base-uncased \
  --negative_file .data/msmarco_passage/raw/train.negatives.tsv \
  --qrels data/msmarco_passage/raw/qrels.train.tsv \
  --queries data/msmarco_passage/raw/train.query.txt \
  --collection data/msmarco_passage/raw/corpus.tsv \
  --save_to ./msmarco_passage/newprocess/train_data/${train_data_name} \

if [ $? -ne 0 ]; then
    echo "build execution failed, script terminated."
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --training_mode ${training_mode} \
  --output_dir ${output_dir} \
  --model_name_or_path ./model/bert-base-uncased \
  --save_steps ${save_steps} \
  --logging_steps 500 \
  --train_dir ${train_data_dir} \
  --fp16 \
  --per_device_train_batch_size ${pd_tbs} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --dataloader_num_workers 2 \
  --train_n_passages 1 \
  --q_max_len ${q_max_len} \
  --p_max_len ${p_max_len}


if [ $? -ne 0 ]; then
    echo "tevatron.driver.train execution failed, script terminated."
    exit 1
fi


model_name=${train_data_name}
model_dir=./msmarco_passage/newmodels/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
save_path=./msmarco_passage/newresults/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
gpu_id=0


config_file="${model_dir}/config.json"
while [ ! -f "$config_file" ]; do
    echo "Waiting for the model configuration file to be generated..."
    sleep 10
done


CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --p_max_len 128 \
  --per_device_eval_batch_size 512 \
  --encode_in_path ./output/corpus_128 \
  --encoded_save_path ${save_path}/corpus_128.pt


if [ $? -ne 0 ]; then
    echo "Corpus encoding execution failed, script terminated."
    exit 1
fi

query_dir=./output
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 512 \
  --encode_in_path ${query_dir}/queries.dev.small.json \
  --encoded_save_path ${save_path}/queries.dev.small.pt


if [ $? -ne 0 ]; then
    echo "Query encoding execution failed, script terminated."
    exit 1
fi


index_type=Flat
index_dir=${save_path}/${index_type}

python -m tevatron.faiss_retriever \
  --query_reps ${save_path}/queries.dev.small.pt \
  --passage_reps ${save_path}/corpus_128.pt \
  --index_type ${index_type} \
  --batch_size 16 \
  --depth 1000 \
  --save_ranking_file ${index_dir}/queries.dev.small.top1k.run.txt \
  --save_index \
  --save_index_dir ${index_dir}


if [ $? -ne 0 ]; then
    echo "tevatron.faiss_retriever execution failed, script terminated."
    exit 1
fi


python test.py data/msmarco_passage/raw/qrels.dev.tsv msmarco_passage/newresults/${model_name}/bert_base_q32p128_pdbs128_lr5e-6_ep4/Flat/queries.dev.small.top1k.run.txt


if [ $? -ne 0 ]; then
    echo "variations_avg_tt_test.py execution failed, script terminated."
    exit 1
fi

echo "All steps executed successfully."    