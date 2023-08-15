from peft import LoraConfig, AdaLoraModel, LoraModel, get_peft_model
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer
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
from LoRA_Ori import seed_everything, TextDataset
from torch.optim import lr_scheduler

#定义dataloader类，用于迭代的遍历数据集 —— 继承Dataset类，并重写__len__()和__getitem__()方法
class ContrastiveDataset(Dataset):
    def __init__(self, filename) -> None:
        super().__init__()
        with open(filename, 'r') as file :
            lines = file.read().splitlines()
            q_c_a_line = [line.split('\t')[0] for line in lines]
            q_c_line = [line.split('answer:')[0] for line in q_c_a_line]
            self.labels = [line.split('\t')[1].strip() for line in lines]
            self.positive_lines = q_c_line
            self.answers    = [line.split('answer:')[1]  for line in q_c_a_line]
            self.questions  = [line.split('context:')[0] for line in q_c_a_line]
            self.contexts   = [line.split('context:')[1] for line in q_c_line]
    def __len__(self):
        return len(self.positive_lines)
    def __getitem__(self, index):
        return self.positive_lines[index] , self.labels[index], self.questions[index], self.contexts[index], self.answers[index]

def contrastive_sampling(batch):
    samples, labels, questions, contexts, answers = zip(*batch)
    batch_size = len(samples)
    negative_indices = torch.randperm(batch_size)
    pos_samples = []
    negative_samples = []
    for i in range(len(samples)):
        #pos_sample =  samples[i]+ "From the context the question can be replied by " + labels[i]
        pos_answer = answers[i]
        pos_question = questions[i]
        #pos_sample = "from the context:" + contexts[i] +" and the answer:" + answers[i] + "the " + questions[i] + " can be answered by " + labels[i]
        pos_sample = questions[i] + "answer:" + answers[i] +"the answer to the question given the context is" + labels[i]
        pos_samples.append(pos_sample)
    for i in negative_indices:
        #negative_sample =samples[i]+"From the context the question can be replied by"+ labels[i]
        
        #negative_sample =samples[i]+"Answer the yes/no/maybe question by reasoning step-by-step:"+ labels[i]
        negative_sample = questions[i] + "context:" + contexts[i] + "the answer to the question given the context is"+ labels[i]
        negative_samples.append(negative_sample)
    #print(samples)
    return samples, labels, pos_samples, negative_samples

def contrastive_loss(ori_hiddent_state, positive_pairs, negative_pairs, temperature=0.1):
    """
    Args:
    positive_pairs: tensor of size (N, D), where N is the batch size and D is the dimension of the vector. 
                    Represents the vectors to be compared positively.
    negative_pairs: tensor of size (N, M, D), where M is the number of negative samples.
    temperature: scalar controlling the temperature of the softmax.

    Returns:
    scalar representing the contrastive loss.
    """
    # Compute similarity scores for positive pairs
    pos_scores = torch.einsum('nd,nd->n', ori_hiddent_state, positive_pairs)  # shape: (N,)
    
    # Compute similarity scores for negative pairs
    neg_scores = torch.einsum('nd,md->nm', ori_hiddent_state, negative_pairs)  # shape: (N, M)
    
    # Concatenate pos_scores and neg_scores
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # shape: (N, M+1)
    
    # Create labels for softmax
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(ori_hiddent_state.device)  # shape: (N,)
    
    # Compute loss
    loss = F.cross_entropy(logits / temperature, labels)
    
    return loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank= rank, world_size= world_size)

def cleanup():
    dist.destroy_process_group()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = './CL_LoRA_T5_models/CL_r16_bs8_model_a32_e100.pth'
model_dir = './CL_Flan_T5_large_models/'
output_model_name = 'qc_CL_r4_bs6_model_a8_e100_mixALL.pth'
model_name = 'google/flan-t5-large'
#train_path = '../../data/PubMedQA/raw/train.tsv'
train_path = '../../data/PubMedQA/T5_lora_data/train_mix_big.tsv'
valid_path = '../../data/PubMedQA/raw/valid.tsv'

train_epoch = 100
bs = 2
optimizer_lr = 1e-3
optimizer_weight_decay = 1e-6
LoRA_r = 4
LoRA_alpha = 8
LoRA_dropout = 0.3
factor = 0.5
patience = 10
CL_temperature = 0.2

#定义LoRA配置，实例化一个LoraConfig类的对象来得到LoRA模块的配置
config = LoraConfig(
    peft_type="LORA",
    task_type="SEQ_2_SEQ_LM",
    r = LoRA_r,
    lora_alpha= LoRA_alpha,
    target_modules=["q","v"],
    lora_dropout= LoRA_dropout
)

# def Train_LoRA_CL():
def Train_LoRA_CL(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    setup(rank, world_size)
    labels = ["yes","no","maybe"]
    le=LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    print(encoded_labels)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_hidden_states=True)

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    model.config
    # 使用get_peft_model得到定义好的模型，传入参数为基模型和peft的配置文件
    lora_model = get_peft_model(model, peft_config=config) 

    #lora_model=torch.load(checkpoint_path)
    lora_model.print_trainable_parameters()

    lora_model.to(rank)
    lora_model = DDP(lora_model, device_ids=[rank])


    output_path = model_dir
    output_name = output_model_name
    batch_size = bs
    num_epochs = train_epoch
    #使用构建正例负例的dataset，创建dataloader实例时，传入负责构建正负例的collate_fn方法
    train_dataset = ContrastiveDataset(train_path)
    valid_dataset = TextDataset(valid_path)
    #dataloader每条输出  pos_sample, label, neg_sample
    # train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn=contrastive_sampling, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)
    #支持分布式并行得dataloader
    train_dist_sampler = distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    valid_dist_sampler = distributed.DistributedSampler(
        valid_dataset,
        num_replicas= world_size,
        rank=rank
    )
    train_dist_dataloader = DataLoader(train_dataset, batch_size= batch_size, sampler= train_dist_sampler, collate_fn=contrastive_sampling)
    valid_dist_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler= valid_dist_sampler)

    # lora_model.to(rank)
    # lora_model = DDP(lora_model, device_ids=[rank])
    #lora_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lora_model.parameters(), lr= optimizer_lr, weight_decay= optimizer_weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    # 开始训练
    min_loss = 10000
    for epoch in range(num_epochs):
        # Set the DistributedSampler as epoch begins
        train_dist_sampler.set_epoch(epoch)
        lora_model.train()
        print(f'Epoch: {epoch}/{num_epochs}')
        epoch_loss = 0
        batch_count = 0

        for inputs, labels, pos_inputs, neg_inputs  in tqdm(train_dist_dataloader):
            # print(len(inputs))
            # print(inputs)
            # print('pos:'+'__'*100)
            # print(pos_inputs)
            # print('neg:'+'__'*100)
            # print(neg_inputs)
            encoding = tokenizer(list(inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            pos_encoding = tokenizer(list(pos_inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
            neg_encoding = tokenizer(list(neg_inputs), return_tensors='pt',padding=True, truncation=True, max_length=512).to(device)
            pos_input_ids = pos_encoding['input_ids'].to(device)
            neg_input_ids = neg_encoding['input_ids'].to(device)
            pos_attention_mask = pos_encoding['attention_mask'].to(device)
            neg_attention_mask = neg_encoding['attention_mask'].to(device)

            labels = torch.tensor([le.transform([label])[0] for label in labels]).to(input_ids.device)
            
            labels = labels.unsqueeze(-1)

            outputs = lora_model(input_ids, attention_mask=attention_mask, output_hidden_states=True,labels=labels)
            pos_outputs = lora_model(pos_input_ids, attention_mask=pos_attention_mask, output_hidden_states=True,labels=labels)
            neg_outputs = lora_model(neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True,labels=labels)

            decoder_last_hidden_state = outputs.decoder_hidden_states[-1].squeeze(1)
            decoder_last_hidden_pos = pos_outputs.decoder_hidden_states[-1].squeeze(1)
            decoder_last_hidden_neg = neg_outputs.decoder_hidden_states[-1].squeeze(1)
            #print("dimentions:", decoder_last_hidden_state.shape)
            #print(decoder_last_hidden_pos.shape)
            #print(decoder_last_hidden_neg.shape)
            
            cl_loss = contrastive_loss(decoder_last_hidden_state, decoder_last_hidden_pos, decoder_last_hidden_neg, temperature=CL_temperature)
            task_loss = outputs.loss
            loss = cl_loss + task_loss

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
        if valid_loss < min_loss:
            min_loss  = valid_loss
            print('*'*40 + f'saving model in epoch {epoch+1}')
            torch.save(lora_model, output_path + output_name)

        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.5f}, Validation Loss: {valid_loss:.5f})')
        print(f'current_lr:{current_lr}')
        scheduler.step(valid_loss)
    #cleanup()
    ori_model = lora_model.module
    torch.save(ori_model.state_dict(), output_path + output_name)


if __name__ == "__main__":
    seed_everything(42)
    world_size = torch.cuda.device_count()
    print("world_size:",world_size)
    torch.multiprocessing.spawn(Train_LoRA_CL, args = (world_size,), nprocs=world_size, join=True)
    #Train_LoRA_CL()




