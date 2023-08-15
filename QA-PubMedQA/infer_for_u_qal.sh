# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_DIR=../../checkpoints/QA-PubMedQA-BioGPT-phase2-l_qal
MODEL=checkpoint_avg.pt
DATA_DIR=${PWD}/../../data/PubMedQA/ansis-bin
BASE_DATA_DIR=${DATA_DIR%/*}
BIN_DATA_DIR=${DATA_DIR##*/}
DATA_PREFIX=${BIN_DATA_DIR%-*}
RAW_DATA_DIR=${BASE_DATA_DIR}/BioGPT_pubmedqa/u_qa_tvt9_1_0
OUTPUT_FILE=generate_uqa_tem_${MODEL}

INPUT_FILE=${RAW_DATA_DIR}/${DATA_PREFIX}_train.tok.bpe.x
OUTPUT_FILE=${MODEL_DIR}/${OUTPUT_FILE}
GOLD_FILE=${RAW_DATA_DIR}/test_with_label.tsv

# average checkpoints
if [ ! -f "${MODEL_DIR}/${MODEL}" ]; then
    python ../../scripts/average_checkpoints.py --inputs=${MODEL_DIR} --output=${MODEL_DIR}/${MODEL} --num-epoch-checkpoints=5
fi

# inference
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Begin inferencing ${INPUT_FILE} using ${MODEL_DIR}/${MODEL}"
    python ../../inference.py --data_dir=${DATA_DIR} --model_dir=${MODEL_DIR} --model_file=${MODEL} --src_file=${INPUT_FILE} --output_file=${OUTPUT_FILE}  --beam=5
fi

# debpe
sed -i "s/@@ //g" ${OUTPUT_FILE}
# detok
perl ${MOSES}/scripts/tokenizer/detokenizer.perl -l en -a < ${OUTPUT_FILE} > ${OUTPUT_FILE}.detok
# postprocess
#CUDA_VISIBLE_DEVICES=0  python postprocess.py ${OUTPUT_FILE}.detok

