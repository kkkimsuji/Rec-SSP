import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

def get_bert_embeddings(text_list, model_name='bert-base-uncased', batch_size=1):
    """
    Extracts BERT [CLS] embeddings for a list of review texts.
    As per the Rec-SSP paper, the [CLS] token captures deep semantic representations[cite: 97, 175].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    # Optimization: Process only unique reviews to save computation time
    unique_texts = list(set(text_list))
    text_to_vec = {}
    
    print(f"Extracting embeddings for {len(unique_texts)} unique reviews on {device}...")
    
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="BERT Embedding"):
        batch = unique_texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token output as the feature vector (Eq 1) [cite: 178, 371]
            vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        for text, vec in zip(batch, vecs):
            text_to_vec[text] = vec
            
    return text_to_vec