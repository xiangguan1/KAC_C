import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from torch.nn import MultiheadAttention
from skimage.filters import threshold_otsu
import argparse
import os

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

class LLMCVGInference(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super(LLMCVGInference, self).__init__()
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA model
        self.qwen = PeftModel.from_pretrained(base_model, model_path)
        self.qwen.eval()
        
        # Get hidden size
        hidden_size = base_model.config.hidden_size
        
        # Load additional layers
        self.cross_attn = MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=0.1).to(device)
        self.filter_1 = nn.Linear(hidden_size, hidden_size//2).to(device)
        self.filter_2 = nn.Linear(hidden_size//2, 11).to(device)
        
        # Load additional layers weights
        additional_layers_path = os.path.join(model_path, 'additional_layers.pth')
        if os.path.exists(additional_layers_path):
            checkpoint = torch.load(additional_layers_path, map_location=device)
            self.cross_attn.load_state_dict(checkpoint['cross_attn_state_dict'])
            self.filter_1.load_state_dict(checkpoint['filter_1_state_dict'])
            self.filter_2.load_state_dict(checkpoint['filter_2_state_dict'])
        
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
    
    def ske_generate(self, filter_fact):
        format_prefix = f"In summary, this legal document belongs to the category"
        acu_combined_text = f"{filter_fact}\n{format_prefix}"
        fact = self.tokenizer(acu_combined_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.qwen.generate(
                **fact, 
                max_new_tokens=100, 
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        outputs = self.tokenizer.batch_decode(outputs[:,fact["input_ids"].size(1):], skip_special_tokens=True)[0]

        pattern = r"of ([^.]*)"  
        match = re.search(pattern, outputs)  
        label = match.group(1) if match else "unknown"

        return label

def get_metrics(predictions, targets):
    return {
        'acc': metrics.accuracy_score(targets, predictions),
        'p': metrics.precision_score(targets, predictions, average='macro', zero_division=1.0),
        'r': metrics.recall_score(targets, predictions, average='macro', zero_division=1.0),
        'f1': metrics.f1_score(targets, predictions, average='macro', zero_division=1.0)
    }

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            fact = batch['fact'][0]
            label = batch['label'][0]
            knowledge = batch['knowledge'][0]
            
            filter_fact, _ = model.filter_token(fact, knowledge, label)
            prediction = model.ske_generate(filter_fact)
            
            predictions.append(prediction)
            targets.append(label)
    
    return get_metrics(predictions, targets)

def main():
    parser = argparse.ArgumentParser(description="Test LLMCVG model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test_file', type=str, required=True, help='Test file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_dataset = CustomDataset(args.test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = LLMCVGInference(args.model_path, device)
    
    # Evaluate
    print("Starting evaluation...")
    metrics = evaluate(model, test_dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Precision: {metrics['p']:.4f}")
    print(f"Recall: {metrics['r']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("="*50)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults saved to {results_file}")

if __name__ == '__main__':
    main()
