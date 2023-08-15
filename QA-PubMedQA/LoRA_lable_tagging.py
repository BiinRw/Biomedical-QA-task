import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from LoRA_Ori import TextDataset,seed_everything
from torch.utils.data import DataLoader, Dataset
import json
from peft import LoraConfig, AdaLoraModel, LoraModel, get_peft_model
import numpy as np

seed_everything(42)
class TextDataset(Dataset):
    def __init__(self, filename) -> None:
        super().__init__()
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
            self.lines = [line.split('a_label:')[0] for line in lines]
            #self.labels = [line.split('\t')[1] for line in lines]
            # self.qc_lines = [line.split('answer:')[0] for line in lines]
            # self.q_lines = [line.split('context:')[0] for line in lines]
            # self.c_lines = [line.split('context:')[1] for line in self.qc_lines]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index):
        return self.lines[index]+ 'the answer to the question given the context is '

def output_decoded_results(decoded_labels, output_path):
    with open(output_path, 'w', encoding='UTF-8') as output_f:
        for item in decoded_labels:
            output_f.write(item)
            output_f.write('\n')
            
config = LoraConfig(
    peft_type="LORA",
    task_type="SEQ_2_SEQ_LM",
    r = 16,
    lora_alpha= 32,
    target_modules=["q","v"],
    lora_dropout= 0.1
)

output_path='./results_processing/persudo_results_aqcal.tsv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path ='./Flan_T5_large_models/qca_prompt(biogpt)_r4_bs6_model_a8_e6_mix.pth'
'加载模型本身'
lora_model = torch.load(model_path)
lora_model.to(device)

'加载模型参数'
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", output_hidden_states=True)
# lora_model = get_peft_model(model, peft_config= config)
# lora_model.load_state_dict(torch.load(model_path))
# lora_model.to(device)

print(type(lora_model))
total_correct = 0
total_samples = 0

labels = ["yes","no","maybe"]
le=LabelEncoder()
encoded_labels = le.fit_transform(labels)
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
test_dataset = TextDataset('../../data/PubMedQA/T5_lora_data/a_qca/train.tsv')
test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle=False)

lora_model.eval()
input_list = []
output_results = []
with torch.no_grad():
    for inputs in test_dataloader:
        input_list.extend([line.split('the answer to the question given the context is')[0] for line in inputs])
        encoding = tokenizer(list(inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        #labels = torch.tensor([le.transform([label])[0] for label in labels]).to(input_ids.device)

        #解码器的输入
        decoder_input_ids = torch.full((input_ids.shape[0],1), tokenizer.pad_token_id, dtype=torch.long).to(device)
        outputs = lora_model(input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids)
        predictions = torch.argmax(outputs.logits, dim = -1)
        
        predictions = predictions.squeeze(1)
        prediction_in_cpu = predictions.to('cpu')
        print("predictions :",predictions)
        #print("ground_truth:",labels)
        pred_numpy = prediction_in_cpu.numpy()
        for element in pred_numpy:
            if element in {2,1,0}:
                
                decoded_labels = le.inverse_transform(np.array([element]))
                output_results.extend(decoded_labels)
                print("decoded_labels:",decoded_labels)
            else:
                print("element:",element)
                output_results.append('failed_label')
            
if len(input_list) != len(output_results):
    print("len(input_list) != len(output_results)")
else:
    persudo_results = []
    for i in range(len(input_list)):
        persudo_line = input_list[i] + "\t "+ output_results[i]
        persudo_results.append(persudo_line)
    
    output_decoded_results(persudo_results, output_path)
        # print(prediction_in_cpu.numpy())
        # print("Decoded labels:",decoded_labels)
        # total_correct += (predictions==labels).sum().item()
        # total_samples += labels.size(0)
# accuracy = total_correct / total_samples
# print("total_correct:",total_correct)
# print("total_samples:",total_samples)
# print('Test Accuracy:', accuracy)

# output_decoded_results(output_results)

