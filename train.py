import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import huggingface_hub
from peft import LoraConfig, get_peft_model
import re
from torch.utils.data import DataLoader, Dataset
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
import argparse

class CustomDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'fact': item['fact'],
            'label': item['label'],
            'knowledge': item['knowledge']
        }

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
    def __init__(self, hidden_size, device="cpu", model_name="xxx", r=8):
        super(LLMCVG, self).__init__()
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=r, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        self.device = device
        
        self.cross_attn = MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=0.1).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
        self.qwen = get_peft_model(model, self.lora_config)

        self.filter_1 = nn.Linear(hidden_size, hidden_size//2).to(device)
        self.filter_2 = nn.Linear(hidden_size//2, 11).to(device)

        self.ls = nn.CrossEntropyLoss()
    
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
        sentence_embeddings = sum_embeddings / sum_mask  
        
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

        acc_embeddings = acc_embeddings.mean(dim=1)

        sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1) 
        acc_embeddings = F.normalize(acc_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.mm(sentence_embeddings_norm, acc_embeddings.T)  
        
        sentence_similarities = similarity_matrix.mean(dim=1)  
        

        threshold = self.find_optimal_threshold(sentence_similarities)
        important_indices = torch.where(sentence_similarities > threshold)[0]

        final_text = self.get_final_sentences_mask(sentences, important_indices)
            
        return final_text, pres_charges
        
    def forward(self, filter_fact, target):

        format_prefix = f"In summary, this legal document belongs to the category of {target}." + self.tokenizer.eos_token

        acu_combined_text = f"{filter_fact}\n{format_prefix}"
        cvg_ids = self.tokenizer(acu_combined_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        cvg_embs = self.qwen.base_model.model.model.embed_tokens(cvg_ids["input_ids"])

        fact_ids = self.tokenizer(filter_fact, return_tensors="pt", padding=True, truncation=True)["input_ids"]

        fact_length = fact_ids.size(1)  

        cvg_labels = cvg_ids["input_ids"].clone()
        cvg_labels[:, :fact_length] = -100

        outputs_acu = self.qwen(inputs_embeds=cvg_embs, attention_mask=cvg_ids["attention_mask"], labels=cvg_labels, return_dict=True, output_hidden_states=True)

        acu_loss = outputs_acu.loss
        
        return acu_loss

    def save_pretrained(self, output_dir):
        self.qwen.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save({
            'filter_1_state_dict': self.filter_1.state_dict(),
            'filter_2_state_dict': self.filter_2.state_dict(),
            'cross_attn_state_dict': self.cross_attn.state_dict()
        }, os.path.join(output_dir, 'additional_layers.pth'))

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()  
    epoch_loss_issue, epoch_loss_pres_issues = 0, 0
    iters = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        
        fact = batch['fact'][0]
        label = batch['label'][0]
        knowledge = batch['knowledge'][0]
        
        filter_fact, pres_charges = model.filter_token(fact, knowledge, label)
        issue_loss = model.forward(filter_fact, label)
        
        target = torch.tensor([label], dtype=torch.long).to(device)
        pres_issue_loss = model.ls(pres_charges, target)
        
        total_loss = issue_loss + pres_issue_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss_issue += issue_loss.item()
        epoch_loss_pres_issues += pres_issue_loss.item()
        iters += 1

    print(f"Epoch {epoch}: LLM Loss: {epoch_loss_issue / iters:.4f}, MLP Loss: {epoch_loss_pres_issues / iters:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train LLMCVG model")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--hidden_size', type=int, required=True, help='Hidden size of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--fact', action='store_true', help='Use fact data')
    parser.add_argument('--label', action='store_true', help='Use label data')
    parser.add_argument('--knowledge', action='store_true', help='Use knowledge data')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--low_rank_training', action='store_true', help='Use LoRA training')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--r', type=int, default=8, help='LoRA rank')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Login to HuggingFace Hub if token exists
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(hf_token)
    
    # Load dataset
    dataset = CustomDataset(args.dataset_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = LLMCVG(
        hidden_size=args.hidden_size,
        device=device,
        model_name=args.model_name_or_path,
        r=args.r
    ).to(device)
    
    if args.bf16:
        model = model.bfloat16()
    
    # Freeze base model parameters, only train LoRA and linear layers
    for name, param in model.named_parameters():
        if "lora" not in name and "qwen" in name:
            param.requires_grad = False
            if "lm_head" in name:
                param.requires_grad = True
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_f1 = 0
    patience_count = 0
    patience = 15
    
    for epoch in range(args.num_train_epochs):
        train_epoch(model, dataloader, optimizer, device, epoch)
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        
        # Simple early stopping (in practice, you'd want to evaluate on validation set)
        # For now, just save every epoch
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Save final model
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
