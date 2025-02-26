# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os


# Setting seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if not os.path.exists('graphs'):
    os.makedirs('graphs')

# hyperparameters to control
EMBED_DIM = 512
PRETRAINED_MODEL = 'distilbert-base-uncased' #'bert-base-uncased'
POOLING_STRAGEY = 'mean'
BATCH_SIZE = 8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

''' ~~~~~~ Part 1: Main Sentence Transformer model ~~~~~~'''
class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embedding_dim=512, pooling_strategy='mean'):
        """
        This class holds the core components of the model architecture.
        Args:
            model_name (str): Pre-trained transformer model name
            embedding_dim (int): Dimension of the sentence embeddings
            pooling_strategy (str): Strategy for pooling token embeddings ('mean', 'max')
        """
        super(SentenceTransformer, self).__init__()
        
        # Load the pre-trained transformer model and tokenizer
        if 'distilbert' not in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.transformer = BertModel.from_pretrained(model_name)
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.transformer = DistilBertModel.from_pretrained(model_name)
        # Output dimension of the transformer
        self.transformer_dim = self.transformer.config.hidden_size
        #linear layer projects the transformer dim into our desired dim no matter which transformer model we use
        self.projection = nn.Linear(self.transformer_dim, embedding_dim)
        # Pooling strategy
        self.pooling_strategy = pooling_strategy
        # Layer normalization for consistency across the embedding layer
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        """ 
        Forward pass through the model
        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer
            attention_mask (torch.Tensor): Attention mask for padding where dim is (batch_size, seq_len) containing 1s for actual tokens and 0s for padding tokens.
        Returns:
            torch.Tensor: Sentence embedding
        """
        # Get transformer outputs
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Get the hidden states from the last layer
        hidden_states = transformer_output.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Apply pooling strategy
        if self.pooling_strategy == 'max': # takes max val for each dim across all token embed in a sentence
            pooled_output, _ = torch.max(hidden_states * attention_mask.unsqueeze(-1), dim=1)
        else:  # mean pooling takes avg of all token embed
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            pooled_output = sum_embeddings / sum_mask
            
        # Project to embedding dimension and normalize
        embedding = self.projection(pooled_output)
        embedding = self.layer_norm(embedding) # (batch_size, embedding_dim)
        # L2 normalization ensures that all embeddings have unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def encode(self, sentences, batch_size=BATCH_SIZE, device=DEVICE):
        """
        Encode a list of sentences into embeddings
        Args:
            sentences (List[str]): List of sentences to encode
            batch_size (int): Batch size for encoding
            device (str): Device to run the model on ('cpu' or 'cuda')
        Returns:
            np.ndarray: Matrix of sentence embeddings
        """
        self.eval()
        self.to(device)
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size] # Get a batch of sentences
            # Tokenize the batch
            encoded_input = self.tokenizer(batch_sentences, 
                                          padding=True, 
                                          truncation=True, 
                                          return_tensors='pt',
                                          )
            
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            # Compute embeddings
            with torch.no_grad():
                embeddings = self.forward(input_ids, attention_mask) # (batch_size, embedding_dim)
            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings) # (num_sentences, embedding_dim)
        
        return all_embeddings


''' ~~~~~~ Part 2: Test model and Visualize ~~~~~~'''
def test_base_model():
    """
    Testng base model with sample sentences
    """
    # Sample sentences
    sentences = [
        "The cat sat on the mat.",
        "The weather is nice today.",
        "A feline was resting on a rug.",
        "The sun is shining outside.",
        "It's a beautiful sunny day outside.",
        "I love working with transformers!",
    ]
    
    # Initialize the model
    model = SentenceTransformer(model_name=PRETRAINED_MODEL, embedding_dim=EMBED_DIM, pooling_strategy=POOLING_STRAGEY)
    # Encode sentences
    embeddings = model.encode(sentences)
    
    print(f"Embedding shape: {embeddings.shape}")
    print("Sample embedding:")
    print(embeddings[0][:10])
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print("\nSimilarity Matrix:")
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='magma')
    plt.colorbar()
    plt.title('Sentence Similarity Matrix')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Index')

    # Add sentence labels
    sentence_labels = [f"S{i+1}" for i in range(len(sentences))]
    plt.xticks(range(len(sentences)), sentence_labels)
    plt.yticks(range(len(sentences)), sentence_labels)
    
    # Add similarity values
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            plt.text(j, i, f"{similarity_matrix[i][j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if similarity_matrix[i][j] < 0.9 else "black")
    
    plt.savefig(f'graphs/TASK1_matrix_{POOLING_STRAGEY}_{PRETRAINED_MODEL}.png', bbox_inches='tight')
    plt.show()
    
test_base_model()