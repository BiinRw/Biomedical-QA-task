import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from LoRA_Ori import TextDataset,seed_everything
from torch.utils.data import DataLoader
import json
from LoRA_CL import config
from peft import LoraConfig, AdaLoraModel, LoraModel, get_peft_model
import numpy as np
import time 

def output_decoded_results(decoded_labels):
    with open('./results_processing/output_results.json', 'w', encoding='UTF-8') as output_f:
        for item in decoded_labels:
            json.dump(item, output_f)
            output_f.write('\n')

def reranking(top_probs, top_indxs):
    variance = np.var(top_probs, axis=1)
    #print(variance)
    return variance
def tensor_dist(tensor_1, tensor_2):
    # tensor_1 = tensor_1.to('cpu').numpy()
    # tensor_2 = tensor_2.to('cpu').numpy()
    abs_diff = torch.abs(tensor_1 - tensor_2)
    squared_diff = torch.pow(tensor_1 - tensor_2, 2)
    relative_diff = torch.abs((tensor_1-tensor_2)/(torch.abs(tensor_1)+1e-6))
    return abs_diff, squared_diff, relative_diff
def softmax(numpy_array):
    e_array = np.exp(numpy_array - np.max(numpy_array, axis=1, keepdims=True))
    return e_array / e_array.sum(axis=1, keepdims=True)

def abs_dist_logits(yes_logits, no_logits, maybe_logits, threshold = 1.0):
    size = len(yes_logits)
    yes_no_abs_dist = torch.abs(torch.sub(yes_logits, no_logits))
    yes_maybe_abs_dist = torch.abs(torch.sub(yes_logits, maybe_logits))
    no_maybe_abs_dist = torch.abs(torch.sub(no_logits, maybe_logits))
    pred = []
    max_tensor = torch.max(torch.max(yes_logits,no_logits), maybe_logits)
    pred = torch.where(max_tensor == yes_logits, torch.tensor(2),
                       torch.where(max_tensor == no_logits, torch.tensor(1), torch.tensor(0)))
    pred =pred.tolist()

    combined = torch.stack((yes_logits, no_logits,maybe_logits))
    mean = combined.mean(dim = 0)
    std_dev = combined.std(dim=0)
    print("mean:", mean)
    print("std_dev:", std_dev)
    yes_standardized = (yes_logits -mean)/std_dev
    no_standardized = (no_logits -mean)/std_dev
    maybe_standardized = (maybe_logits -mean)/std_dev
    print("yes_standardized:", yes_standardized)
    print("no_standardized:", no_standardized)
    print("maybe_standardized:", maybe_standardized)
    # print("yes_no_abs_dist:",yes_no_abs_dist)
    # print("yes_maybe_abs_dist:",yes_maybe_abs_dist)
    # print("no_maybe_abs_dist:",no_maybe_abs_dist)
    for indext in range(size):
        # if yes_no_abs_dist[indext].item()<=threshold and yes_maybe_abs_dist[indext].item()<=threshold \
        #     and no_maybe_abs_dist[indext].item()<=threshold:
        # if  yes_maybe_abs_dist[indext].item()<=threshold \
        #     and no_maybe_abs_dist[indext].item()<=threshold:
        # if (yes_standardized[indext] >0 and no_standardized[indext] >0 and maybe_standardized[indext] <0 ) or \
        # (yes_standardized[indext] <0 and no_standardized[indext] >0 and maybe_standardized[indext] >0 ) or \
        # (yes_standardized[indext] >0 and no_standardized[indext] <0 and maybe_standardized[indext] >0 ):
            pred[indext] = 0
    return np.array(pred)
            

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'google/flan-t5-large'
#model_path ='../BioASQ/Few_Shots_models/qca_T5large_5shots.pth'
#model_path ='./Flan_T5_large_models/qc_prompt(bioGPT)_r4_bs6_model_a8_e2_MixALL.pth'
model_path = './LoRA_T5_models/qc_biogpt_r4_bs6_a8.pth'
'BioASQ test'
#test_data_path ='../../data/BioASQ/splited_qcal_BioASQ_training10b/test.tsv'
'PubmedQA test'
test_data_path ='../../data/PubMedQA/raw/test_with_label.tsv'
'PubmedQA a_qca dataset'
#test_data_path = '../../data/PubMedQA/T5_lora_data/a_qca/train.tsv'

'加载模型本身'
lora_model = torch.load(model_path)
lora_model.to(device)

'加载模型参数'
#model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", output_hidden_states=True)
# lora_model = get_peft_model(model, peft_config= config)
# #lora_model.load_state_dict(torch.load(model_path))
# lora_model.to(device)

print(type(lora_model))
print(lora_model)
total_correct = 0
total_samples = 0
optimized_correct = 0

labels = ["yes","no","maybe"]
le=LabelEncoder()
encoded_labels = le.fit_transform(labels)
tokenizer = T5Tokenizer.from_pretrained(model_type)
test_dataset = TextDataset(test_data_path)
test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle=False)

lora_model.eval()

output_results = []
true_labels = []
pred_labels =[]
optimized_pred_labels =[]

start_time = time.time()

with torch.no_grad():
    for inputs, labels in test_dataloader:
        encoding = tokenizer(list(inputs), return_tensors='pt',padding= True, truncation=True, max_length=512).to(device)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = torch.tensor([le.transform([label])[0] for label in labels]).to(input_ids.device)

        #解码器的输入
        decoder_input_ids = torch.full((input_ids.shape[0],1), tokenizer.pad_token_id, dtype=torch.long).to(device)
        outputs = lora_model(input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids)
        #print(outputs.logits[0,0,0])
        #print(outputs.logits.shape)

        predictions = torch.argmax(outputs.logits, dim = -1)
        probs = F.softmax(outputs.logits, dim = -1)
        probs = probs.squeeze(1)

        top_probs, top_idxs = torch.topk(outputs.logits, 3)
        top_probs_np = top_probs.to('cpu').numpy().squeeze(1)
        top_idxs_np = top_idxs.to('cpu').numpy().squeeze(1)
        #print('top_probs:\n',top_probs, "top_idxs:\n", top_idxs)
        #variance = reranking(top_probs, top_idxs)

        top3_probs = softmax(top_probs_np)
        print("top3_probs:",top3_probs)
        maybe_logits = outputs.logits[:,:,0].squeeze(1)
        yes_logits = outputs.logits[:,:,2].squeeze(1)
        no_logits = outputs.logits[:,:,1].squeeze(1)
        optimized_pred = abs_dist_logits(yes_logits=yes_logits, no_logits=no_logits, maybe_logits= maybe_logits)

        yes_no_abs_dist, yes_no_squared_dist, yes_no_relative_dist= tensor_dist(yes_logits, no_logits)
        maybe_yes_abs_dist,  maybe_yes_squared_dist,  maybe_yes_relative_dist= tensor_dist(maybe_logits, yes_logits)
        maybe_no_abs_dist,  maybe_no_squared_dist, maybe_no_relative_dist= tensor_dist(maybe_logits, no_logits)
        # print(f"yes_no_abs_dist :       {yes_no_abs_dist}")
        # print(f"maybe_yes_abs_dist:     {maybe_yes_abs_dist}")
        # print(f"maybe_no_abs_dist:      {maybe_no_abs_dist}")

        # print(f"yes_no_relative_dist:   {yes_no_relative_dist}")
        # print(f"maybe_yes_relative_dist:{maybe_yes_relative_dist}")
        # print(f"maybe_no_relative_dist: {maybe_no_relative_dist}")
        
        # print(f"yes_no_squared_dist:    {yes_no_squared_dist}")
        # print(f"maybe_yes_squared_dist: {maybe_yes_squared_dist}")
        # print(f"maybe_no_squared_dist:  {maybe_no_squared_dist}")
        
        #print ('variance:',variance)
        # print('logits: index 2', yes_logits)
        # print('logits: index 1', no_logits)
        # print('logits: index 0', maybe_logits)
        # print("probs:", probs)
        predictions = predictions.squeeze(1)
        prediction_in_cpu = predictions.to('cpu')
        print("predictions :",prediction_in_cpu.numpy())
        print("ground_truth: ",labels.to('cpu').numpy())
        print("optimized_preds:", optimized_pred)
        pred_labels.extend(prediction_in_cpu.numpy().tolist())
        
        true_labels.extend(labels.cpu().numpy().tolist())
        decoded_labels = le.inverse_transform(prediction_in_cpu.numpy())
        output_results.extend(decoded_labels)
        
        # print(prediction_in_cpu.numpy())
        # print("Decoded labels:",decoded_labels)
        total_correct += (predictions==labels).sum().item()
        total_samples += labels.size(0)
        optimized_correct += (torch.tensor(optimized_pred)==labels.to('cpu')).sum().item()
        optimized_pred_labels.extend(optimized_pred)

accuracy = total_correct / total_samples
opt_accuracy = optimized_correct / total_samples
print("total_correct:",total_correct)
print("total_samples:",total_samples)
print('Test Accuracy:', accuracy)
print('Optimized Accuracy:', opt_accuracy)
macro_f1 = f1_score(true_labels, pred_labels, average='macro')
opt_macro_f1 = f1_score(true_labels, optimized_pred_labels, average='macro')
print("Macro-F1: ",macro_f1)
print("opt_marco-F1: ",opt_macro_f1)
output_decoded_results(output_results)
end_time = time.time()
execution_time = end_time - start_time
print("execution_time:",execution_time)


