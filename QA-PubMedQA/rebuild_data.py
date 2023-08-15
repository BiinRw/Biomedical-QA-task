# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

data_dir=sys.argv[1]


# def build_target_seq(tgt):
#     tgt = 'the answer to the question given the context is ' + tgt + '.'
#     return tgt
def build_target_seq(tgt, data_tpye='ori'):
    if data_tpye == 'ori':
        tgt = 'the answer to the question given the context is ' + tgt + '.'
    if data_tpye == 'qcl':
        tgt = 'From the context, the question can be answered by ' + tgt + '.'
    if data_tpye == 'qal':
        tgt = 'From the answer the question can be replied by ' + tgt + '.'
    if data_tpye == 'cot':
        tgt = 'Let us think step by step, the answer is' + tgt + '.'

    return tgt

def loader(fname, fn,):
    ret = []
    cnt = 0
    file = open(fname)
    
    for line in file:
        if line == '\n':
            continue
        cnt += 1
        sent = line.rsplit('\t',1)
        print(cnt)
        source, target = sent[0].replace('\n', '').strip()+".", sent[1].replace('\n', '').strip()
        print("source:",source)
        print("target:",target)
        source = source.lower()
        target = target.lower()
        if source[-1] == '.':
            #print("fn(target):",fn(target))
            ret.append([source, fn(target)])
        else:
            ret.append([source +'.', fn(target)])

    print(f"{cnt} samples in {fname} has been processed")
    #print("ret:", ret)
    return ret


def dumper(content_list, prefix):
    fw_source = open(prefix + ".x", "w")
    fw_target = open(prefix + ".y", "w")
    
    for ele in content_list:
        print(ele[0], file=fw_source)
        #print("0:",ele[0])
        #print("1:",ele[1])
        print(ele[1], file=fw_target)

    fw_source.close()
    fw_target.close()


def worker(fname, prefix, fn, ):
    ret = loader(fname, fn, )
    dumper(ret, prefix)


for split in ['train', 'valid', 'test']:
    worker(os.path.join(f"{data_dir}", f"{split}.tsv"), os.path.join(f"{data_dir}", f"ansis_{split}"), build_target_seq)
