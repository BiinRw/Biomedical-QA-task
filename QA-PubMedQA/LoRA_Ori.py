from peft import LoraConfig, AdaLoraModel, LoraModel, get_peft_model
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import torch
from torch.utils.data import DataLoader, distributed
#from fairseq.models.transformer_lm import TransformerLanguageModel
#from torchtext.vocab import build_vocab_from_iterator
#from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from mosestokenizer import MosesTokenizer
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from torch.optim import lr_scheduler
import time 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#定义dataloader类，用于迭代的遍历数据集 —— 继承Dataset类，并重写__len__()和__getitem__()方法
class TextDataset(Dataset):
    def __init__(self, filename) -> None:
        super().__init__()
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
            self.lines = [line.split('\t')[0] for line in lines]
            self.labels = [line.split('\t')[1].strip()  for line in lines]
            self.qc_lines = [line.split('answer:')[0] for line in lines]
            self.q_lines = [line.split('context:')[0] for line in lines]
            #self.c_lines = [line.split('context:')[1] for line in self.qc_lines]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index):
        #return self.qc_lines[index]+ 'From the context the question can be replied by ', self.labels[index]
        #return self.lines[index]+ 'From the context the question can be replied by ', self.labels[index]
        
        # BioGPT prompt'0.692' '0.694' '0.732' '0.826_qca'
        return self.qc_lines[index]+ 'the answer to the question given the context is ', self.labels[index]
        #return self.lines[index]+ 'the answer to the question given the context is ', self.labels[index]
        'cq_prompt_r4_bs8_model_a16_e100, 0.654'
        #return 'context:'+ self.c_lines[index]+ self.q_lines[index]+ 'which label in (yes, no, maybe) can best answer the question? ', self.labels[index]
        '0.656'
        #return self.q_lines[index]+ self.c_lines[index] + "the answer to the question is", self.labels[index]
        #the answer to the question given the context is 
        
        #CoT for Flan-T5
        #return self.qc_lines[index]+ 'Answer the yes/no/maybe question by reasoning step-by-step:', self.labels[index]
        '0.686' '73.8'
        # return self.lines[index]+ 'Answer the yes/no/maybe question by reasoning step-by-step:', self.labels[index]

        #return self.qc_lines[index]+ 'The question can be answerd by', self.labels[index]
        #return self.lines[index]+ 'The question can be answerd by', self.labels[index]


#path_names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 't5-base'
checkpoints_path = '../QA-PubMedQA/Flan_T5_large_models/qc_biogpt_r4_bs6_a8_lr1-2_e30.pth'

model_dir = '../QA-PubMedQA/LoRA_T5_models/'
model_name = 'qc_biogpt_r4_bs6_a8.pth'

#train_path ='../../data/BioASQ/splited_qcal_BioASQ_training10b/train_few_shots.tsv'
train_path = '../../data/PubMedQA/raw/train_qcal.tsv'
#train_path = '../../data/PubMedQA/T5_lora_data/train_mix_big.tsv'
#train_path = '../../data/PubMedQA/raw/train_few_shots.tsv'
#train_path = './results_processing/persudo_results_aqcal.tsv'
valid_path = '../../data/PubMedQA/raw/valid_qcal.tsv'
#valid_path = '../../data/BioASQ/splited_qcal_BioASQ_training10b/valid.tsv'
#hyper-parameters
train_epoch = 100
bs = 6
optimizer_lr = 1e-4
optimizer_weight_decay = 1e-7
LoRA_r = 4
LoRA_alpha = 8
LoRA_dropout = 0.1
factor = 0.5
patience = 10
#定义LoRA配置，实例化一个LoraConfig类的对象来得到LoRA模块的配置
config = LoraConfig(
    peft_type="LORA",
    task_type="SEQ_2_SEQ_LM",
    r = LoRA_r,
    lora_alpha= LoRA_alpha,
    target_modules=["q","v"],
    lora_dropout= LoRA_dropout
)


def Train_LoRA_ori():
    labels = ["yes","no","maybe"]
    le=LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    print(encoded_labels)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)   #google/flan-t5-large
    model.config
    # 使用get_peft_model得到定义好的模型，传入参数为基模型和peft的配置文件
    lora_model = get_peft_model(model, peft_config=config) 
    
    ##模型加载
    #lora_model = model
    #lora_model = torch.load(checkpoints_path)
    lora_model.print_trainable_parameters()
    tokenizer = T5Tokenizer.from_pretrained(model_type)

    output_path = model_dir
    output_name = model_name
    batch_size = bs
    num_epochs=train_epoch

    dataset = TextDataset(train_path)
    valid_dataset = TextDataset(valid_path)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    valid_dist_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    lora_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay, )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    # 开始训练
    min_loss = 10000
    lora_model.train()
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs}')
        epoch_loss = 0
        batch_count = 0
        for inputs, labels in tqdm(dataloader):
            #print(inputs)
            #print(labels)
            encoding = tokenizer(list(inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
            input_ids = encoding['input_ids'].to(device)
            
            attention_mask = encoding['attention_mask'].to(device)
            labels = torch.tensor([le.transform([label])[0] for label in labels]).to(input_ids.device)
            
            labels = labels.unsqueeze(-1)

            outputs = lora_model(input_ids, attention_mask=attention_mask,labels=labels)
            loss = outputs.loss
            
            epoch_loss += loss.item()
            batch_count +=1
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = epoch_loss /batch_count
        
        current_lr = optimizer.param_groups[0]['lr']
        #eval模式
        lora_model.eval()
        valid_loss = 0.0
        
        for valid_inputs, valid_labels in tqdm(valid_dist_dataloader):
            with torch.no_grad():
                encoding = tokenizer(list(valid_inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
                valid_inputs_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                valid_labels = torch.tensor([le.transform([label])[0] for label in valid_labels]).to(valid_inputs_ids.device)
                valid_labels = valid_labels.unsqueeze(-1)
                valid_outputs = lora_model(valid_inputs_ids, attention_mask=attention_mask, output_hidden_states=True,labels=valid_labels)
                loss = valid_outputs.loss
                valid_loss += loss.item()
        valid_loss /=len(valid_dist_dataloader)
        #scheduler.step(valid_loss)
        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.5f}, Validation Loss: {valid_loss:.5f})')
        print(f'current_lr:{current_lr}')
        if valid_loss < min_loss:
            min_loss  = valid_loss
            print('*'*40 + f'saving model in epoch {epoch+1}')
            torch.save(lora_model, output_path + output_name)
        scheduler.step(valid_loss)

    #torch.save(lora_model, output_path + output_name)

if __name__ == "__main__":
    # world_size = torch.cuda.device_count()
    # print(world_size)
    # torch.multiprocessing.spawn(Train_LoRA_CL, args = (world_size,), nprocs=world_size, join=True)
    seed_everything(42)
    start_time = time.time()
    Train_LoRA_ori()
    end_time = time.time()
    execution_time = end_time - start_time
    print("execution_time:",execution_time)




    # model = TransformerLanguageModel.from_pretrained(
#     "../../checkpoints/Pre-trained-BioGPT",
#     "checkpoint.pt",
#     "../../data",
#     tokenizer = 'moses',
#     bpe = 'fastbpe',
#     bpe_codes = "../../data/bpecodes",
#     min_len = 100,
#     max_len_b = 1024
# )
# TEXT = Field(sequential =True, tokenizer=model.tokenizer, lower = True )
# LABEL = Field(sequential = True, use_vocab=False)

# datafields = [("text", TEXT), ("label", LABEL)]
# train_data, valid_data, test_data = TabularDataset.splits(
#     path="../../data/PubMedQA/raw",
#     train='train.tsv',
#     validation = 'valid.tsv',
#     test = 'test.tsv',
#     skip_header = True,
#     fields = datafields
# )
# TEXT.build_vocab(train_data, max_size=25000)

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     train_data, valid_data, test_data,
#     batch_size =64,
#     device=device)


# for inputs, labels in enumerate(dataloader):
#     print(inputs)
#     print(labels)

# def loader(fname):
#     ret = []
#     cnt = 0
#     file = open(fname)
#     for line in file:
#         if line == '\n':
#             continue
#         cnt += 1
#         sent = line.rsplit('.',1)
        
#         text, label = sent[0].replace('\n', '').strip()+".", sent[1].replace('\n', '').strip()
#         text = text.lower()
#         label = label.lower()
#         ret.append([text, label])
#     print(f"{cnt} samples in {fname} has been processed")
#     #print("ret:", ret)
#     return ret

# def yield_tokens(data_iter):
#     for text, _ in data_iter:
#         yield tokenizer(text)

# def collate_batch(batch):
#     label_list, text_list = [], []
#     for _text, _label in batch:
#         label_list.append(_label)
#         processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)], dtype=torch.int64)
#         text_list.append(processed_text)
#     return torch.tensor(label_list), pad_sequence(text_list, padding_value = 1.0)


# trian_dataset = loader("../../data/PubMedQA/raw/train.tsv")
# valid_dataset = loader("../../data/PubMedQA/raw/valid.tsv")
# test_dataset = loader("../../data/PubMedQA/raw/test.tsv")

# vocab = build_vocab_from_iterator(yield_tokens(trian_dataset), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])

# train_dataloader = DataLoader(trian_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
# valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
# test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
