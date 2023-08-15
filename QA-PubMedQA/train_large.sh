# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

SAVE_DIR=/mnt/disk2/wbr/BioGPT/checkpoints/QA-PubMedQA-GPT2-Large-BioASQ
mkdir -p ${SAVE_DIR}

CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
    ../../data/PubMedQA/biogpt-large-ansis-bin --save-dir ${SAVE_DIR} \
    --user-dir ../../src \
    --finetune-from-model ../../checkpoints/Pre-trained-BioGPT/Pre-trained-BioGPT-Large/checkpoint.pt \
    --task language_modeling_prompt \
    --arch transformer_lm_prompt_biogpt_large \
    --share-decoder-input-output-embed --decoder-learned-pos \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 --clip-norm 0.0 \
    --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 100 --warmup-init-lr 1e-07 \
    --tokens-per-sample 768 --max-source-positions 1900 --max-target-positions 2048 \
    --max-tokens 2048 --update-freq 8 \
    --skip-invalid-size-inputs-valid-test \
    --max-epoch 100 --keep-last-epochs 5 \
    --learned-prompt 9