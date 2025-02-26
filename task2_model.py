import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer


# Hyperparameters
BATCH_SIZE = 8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embedding_dim=512):
        """
        This class holds the core components of the model architecture along with multitask heads.
        Args:
            model_name (str): Pre-trained transformer model name.
            embedding_dim (int): Dimension of the sentence embeddings.
            pooling_strategy (str): Strategy for pooling token embeddings ('mean', 'max').
        """
        super(MultiTaskSentenceTransformer, self).__init__()
        
        # Load the pre-trained transformer model and tokenizer
        if 'distilbert' not in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.transformer = BertModel.from_pretrained(model_name)
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.transformer = DistilBertModel.from_pretrained(model_name)
        # Output dimension of the transformer
        self.transformer_dim = self.transformer.config.hidden_size
        # Linear layer projects the transformer dim into our desired embedding dim
        self.projection = nn.Linear(self.transformer_dim, embedding_dim)
        # Layer normalization for consistency across the embed layer
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Task A head: Sentence classification (technology or entertainment)
        num_classes = 2
        self.taskA_classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(embedding_dim // 2, num_classes)
            )
            
        #Task B head: Sentiment analysis (negative or positive)
        self.taskB_classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(embedding_dim // 2, num_classes)
            )
        
    def forward(self, input_ids, attention_mask, multitask=True):
        """
        Forward pass through the model.
        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer.
            attention_mask (torch.Tensor): Attention mask for padding. Shape: (batch_size, seq_len) with 1s for actual tokens and 0s for padding.
            multitask (bool): If True, return classification logits for both tasks in addition to the sentence embeddings.
        
        Returns:
            If multitask is False:
                torch.Tensor: Sentence embeddings of shape (batch_size, embedding_dim)
            If multitask is True:
                tuple: (embedding, taskA_logits, taskB_logits)
        """
        # Get transformer outputs
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Get the hidden states from the last layer (batch_size, seq_len, hidden_size)
        hidden_states = transformer_output.last_hidden_state
        
        # Apply mean pooling strategy to obtain a fixed-size sentence representation
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_mask = torch.sum(attention_mask_expanded, dim=1)
        pooled_output = sum_embeddings / sum_mask
            
        # Project to the desired embedding dimension and normalize
        embedding = self.projection(pooled_output)
        embedding = self.layer_norm(embedding)
        # L2 normalization ensures that all embeddings have unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        if multitask:
            # Compute logits for each task using the shared embedding
            taskA_logits = self.taskA_classifier(embedding)
            taskB_logits = self.taskB_classifier(embedding)
            return embedding, taskA_logits, taskB_logits
        
        return embedding
    
    def encode(self, sentences, batch_size=BATCH_SIZE, device=DEVICE):
        """
        Encode a list of sentences into embeddings.
        Args:
            sentences (List[str]): List of sentences to encode.
            batch_size (int): Batch size for encoding.
            device (str): Device to run the model.
        """
        self.eval()
        self.to(device)
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            # Tokenize the batch
            encoded_input = self.tokenizer(batch_sentences, 
                                           padding=True, 
                                           truncation=True, 
                                           return_tensors='pt')
            
            input_ids = encoded_input['input_ids'].to(device) # Shape: (batch_size, seq_len)
            attention_mask = encoded_input['attention_mask'].to(device) # Shape: (batch_size, seq_len)
            
            # Compute embeddings (without multitask classification heads)
            with torch.no_grad():
                embeddings = self.forward(input_ids, attention_mask, multitask=False)
            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings
