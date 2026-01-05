import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import huggingface_hub
from peft import LoraConfig, get_peft_model
import re
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import numpy as np
import os
from sklearn import metrics
from torch.nn import MultiheadAttention
import torch.distributed as dist
import torch.multiprocessing as mp
from skimage.filters import threshold_otsu
from datetime import datetime

huggingface_hub.login("xxx")

def get_metrics(p, t):
    if isinstance(p, torch.Tensor):
        p = p.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()

    return {
        'acc': metrics.accuracy_score(t, p),
        'p': metrics.precision_score(t, p, average='macro', zero_division=1.0),
        'r': metrics.recall_score(t, p, average='macro', zero_division=1.0),
        'f1': metrics.f1_score(t, p, average='macro', zero_division=1.0)
    }


class LLMCVG(nn.Module):
    def __init__(self, hidden_size='model_hidden_size', device="cpu", rand_init=False, 
                 model_name="xxx", use_partial_layers=False, num_layers=4):
        super(LLMCVG, self).__init__()
        self.lora_config = LoraConfig(
            r=256,
            lora_alpha=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # lora_dropout=0.2,
            bias="none"
        )
        self.device = device
        self.use_partial_layers = use_partial_layers
        self.num_layers = num_layers

        # self.self_attn = MultiheadAttention(
        #     embed_dim=hidden_size,
        #     num_heads=8,
        #     dropout=0.1,
        #     batch_first=True  # 使用batch_first格式更易处理
        # ).to(device)
        
        self.cross_attn = MultiheadAttention(embed_dim=hidden_size,num_heads=8,dropout=0.1).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})

        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
        self.qwen = get_peft_model(model, self.lora_config)

        # 分类相关结构
        self.filter_1 = nn.Linear(hidden_size, hidden_size//2).to(device)
        self.filter_2 = nn.Linear(hidden_size//2, 11).to(device)

        self.ls = nn.CrossEntropyLoss()

        self.gen_config = transformers.GenerationConfig(
            max_new_tokens=8,
            min_new_tokens=None,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            typical_p=1.0,
            repetition_penalty=1.176,

            num_return_sequences=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        self.count_split = 0
        self.count_xxx = 0
        self.defendant_null = 0
    
    def find_optimal_threshold(self, similarity_scores):
        similarity_scores = similarity_scores.float().cpu().numpy()
        scores_np = np.array(similarity_scores)
        threshold = threshold_otsu(scores_np)
        return threshold
    
    def get_final_sentences_mask(self, sentences, important_indices):
        n = len(sentences)
        
        include_mask = torch.zeros(n, dtype=torch.bool)
        
        for idx in important_indices:
            start = max(0, idx - 1)
            end = min(n, idx + 2)
            include_mask[start:end] = True
        
        final_indices = torch.where(include_mask)[0]
        final_text = "".join([sentences[i] for i in final_indices.tolist()])
        
        return final_text

    def filter_token(self, fact, knowledge, label):
        fact_text = fact
        
        sentences = [s.strip() for s in re.findall(r'.*?[,.!?]', fact_text) if s.strip()]
        
        if not sentences:
            sentences = [fact_text]
        
        sentence_inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        input_ids = sentence_inputs.input_ids
        
        sentence_embeddings = self.qwen.base_model.model.model.embed_tokens(input_ids)

        attention_mask = sentence_inputs.attention_mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(sentence_embeddings.size())

        sum_embeddings = torch.sum(sentence_embeddings * mask_expanded, dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        sentence_embeddings = sum_embeddings / sum_mask  # [num_sentences, hidden_size]
        
        pres_charges = self.filter_1(sentence_embeddings)
        pres_charges = pres_charges.mean(dim=0, keepdim=True)
        pres_charges = self.filter_2(pres_charges)

        _, top_3_indices = torch.topk(pres_charges, k=3, dim=1)
        top_3_indices = top_3_indices.squeeze() 

        top_3_charges = [label[top_3_indices[i].item()] for i in range(3)]
        knowledge = knowledge[top_3_charges]

        acc_inputs = self.tokenizer(
            knowledge,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        acc_input_ids = acc_inputs.input_ids
        acc_embeddings = self.qwen.base_model.model.model.embed_tokens(acc_input_ids)

        acc_embeddings = acc_embeddings.mean(dim=1)  # [num_acc, hidden_size]

        sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)  # [num_sentences, hidden_dim]
        acc_embeddings = F.normalize(acc_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.mm(sentence_embeddings_norm, acc_embeddings.T)  # [num_sentences, num_acc]
        
        sentence_similarities = similarity_matrix.mean(dim=1)  # [num_sentences]
        

        threshold = self.find_optimal_threshold(sentence_similarities)
        important_indices = torch.where(sentence_similarities > threshold)[0]

        final_text = self.get_final_sentences_mask(sentences, important_indices)
            
        return final_text, pres_charges
        
    def forward(self, filter_fact, target):

        format_prefix = f"In summary, this legal document belongs to the category of {target}." + self.tokenizer.eos_token

        acu_combined_text = f"{filter_fact}\n{format_prefix}"
        cvg_ids = self.tokenizer(acu_combined_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        cvg_embs = self.qwen.base_model.model.model.embed_tokens(cvg_ids["input_ids"])
        # pdb.set_trace()
        # cvg_ids['attention_mask'] = torch.cat(
        fact_ids = self.tokenizer(filter_fact, return_tensors="pt", padding=True, truncation=True)["input_ids"]

        fact_length = fact_ids.size(1)  

        cvg_labels = cvg_ids["input_ids"].clone()
        cvg_labels[:, :fact_length] = -100

        outputs_acu = self.qwen(inputs_embeds=cvg_embs, attention_mask=cvg_ids["attention_mask"], labels=cvg_labels, return_dict=True, output_hidden_states=True)

        acu_loss = outputs_acu.loss
        
        return acu_loss

    def ske_generate(self, filter_fact):

        format_prefix = f"In summary, this legal document belongs to the category"
        acu_combined_text = f"{filter_fact}\n{format_prefix}"
        fact = self.tokenizer(acu_combined_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        outputs = self.qwen.generate(**fact, max_new_tokens=100, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        outputs = self.tokenizer.batch_decode(outputs[:,fact["input_ids"].size(1):], skip_special_tokens=True)[0]

        pattern = r"of ([^.]*)"  
        match = re.search(pattern, outputs)  
        label = match.group(1)

        return label

def train(model, fact, true_label, knowledge, e):
    epoch_loss_issue, epoch_loss_pres_issues = 0, 0
    iters = 0
    model.train()  # 改动
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        filter_fact, pres_charges = model.filter_token(fact, knowledge)
        issue_loss = model.forward(filter_fact, true_label)
        target = torch.tensor(batch['labels'], dtype=torch.long).to(device)
        pres_issue_loss = model.ls(pres_charges, target)
        
        (issue_loss + pres_issue_loss).backward()

        optimizer.step()


        epoch_loss_issue += issue_loss.item()
        epoch_loss_pres_issues += pres_issue_loss.item()
        
        iters += 1

    print(f"Epoch {e}:LLM Loss: {epoch_loss_issue / iters:.2f}, "f"MLP Loss: {epoch_loss_pres_issues / iters:.2f}")

def test(model, fact, true_label):
    model.eval()
    target_issue, pres_issues_list = [], []

    for batch in tqdm(test_dataloader):
        filter_fact, _ = model.filter_token(fact) 

        issue_out = model.ske_generate(filter_fact)
        pres_issues_list.extend([issue_out])


        target_issue.extend([true_label])

    out_issues = get_metrics(pres_issues_list, target_issue)
    print(out_issues)




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 1
    model = LLMCVG(device=device).to(device).bfloat16()
    for name, param in model.named_parameters():
        if "lora" not in name and "qwen" in name:  
            param.requires_grad = False
            if "lm_head" in name:
                param.requires_grad = True

    linear_params = list(model.filter_1.parameters()) + list(model.filter_2.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    train = json.load(open("xxx", 'r', encoding='utf-8'))
    test = json.load(open("xxx", 'r', encoding='utf-8'))


    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


    for e in range(epochs):
        train(model, fact, true_label, knowledge, e)

        out_issues = test(model, fact, true_label)

        print(f"Epoch {e}: issues: {out_issues}")

        rmse_r = out_issues["f1"]

        if rmse_r > best_rmse:
            print("=" * 10 + "BEST EPOCH" + "=" * 10)
            best_rmse = rmse_r
            count = 0
            best_issues = out_issues

        else:
            count += 1

        if count >= 15:
            print("Early stopping because metric not improving.")
            break
        
        print("*" * 10 + "BEST" + "*" * 10)
        print(f"BEST Result => BEST: {best_issues}")

        print("Done.")








