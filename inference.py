# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from src.transformer_lm_prompt import TransformerLanguageModelPrompt
import torch
import time 


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--model_file", type=str, default="checkpoint_last.pt")
parser.add_argument("--src_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--decoding_length", type=int, default=1024)
args, _ = parser.parse_known_args()


def main(args):
    src_inputs = []
    with open(args.src_file) as reader:
        for line in reader:
            src_inputs.append(line.strip())
    
    m = TransformerLanguageModelPrompt.from_pretrained(
        args.model_dir, 
        args.model_file, 
        args.data_dir,
        max_len_b=args.decoding_length,
        max_tokens=12000,)

    print(m.cfg)

    if m.cfg.common.fp16:
        print('Converting to float 16')
        m.half()
    m.cuda()
    m.eval()
    outputs=[]
    for i in range(0,len(src_inputs),50):
        input=src_inputs[i:i+50]
        tmp_outputs = m.sample(input, beam=args.beam, temperature=0.1)
        outputs.extend(tmp_outputs)
        
    with open(f"{args.output_file}", "w", encoding='utf8') as fw:
        for i in range(len(outputs)):
            fw.write(outputs[i] + '\n')


if __name__ == "__main__":
    start_time = time.time()
    torch.cuda.empty_cache()
    main(args)
    end_time = time.time()
    execution_time = end_time - start_time
    print("execution_time:",execution_time)