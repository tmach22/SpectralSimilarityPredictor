import torch
import torch.nn as nn
import pickle
from massformer.massformer import MassFormer
from massformer.args import train_args

class MassFormerEncoder(nn.Module):
    """
    Performs the "head-ectomy" on the pre-trained MassFormer model.
    It accepts a batch dictionary from the collator and returns a graph embedding.
    """
    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__()
        args = train_args()
        args.load(config_path)
        self.full_massformer_model = MassFormer(args.model)

        with open(checkpoint_path, "rb") as f:
            state_dict = pickle.load(f)["state_dict"]
        
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.full_massformer_model.load_state_dict(state_dict)
        self.full_massformer_model.eval() # Set to evaluation mode

    def forward(self, batch_dict: dict) -> torch.Tensor:
        """
        The forward pass now accepts the dictionary of padded tensors.
        """
        # The original MassFormer model's forward pass takes the batch dictionary
        # directly as input. We will call it to get the node embeddings.
        # This is the main part of the encoder.
        node_embeddings = self.full_massformer_model.encoder(batch_dict)
        
        # The pooling layer needs the node embeddings and a way to distinguish
        # the nodes of each graph in the batch. We can create a batch index tensor.
        num_graphs = node_embeddings.size(0)
        num_nodes_per_graph = node_embeddings.size(1)
        batch_index = torch.arange(num_graphs, device=node_embeddings.device).repeat_interleave(num_nodes_per_graph)
        
        # Flatten the node embeddings for the pooling layer
        node_embeddings_flat = node_embeddings.view(-1, node_embeddings.size(-1))

        # Apply the readout/pooling function to get a single graph-level embedding
        graph_embedding = self.full_massformer_model.pool(node_embeddings_flat, batch_index)
        
        return graph_embedding

class SimilarityHead(nn.Module):
    """
    An MLP that takes the combined embeddings and predicts the similarity score.
    """
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # To constrain output between 0 and 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1) # Remove the last dimension

class SiameseSpectralSimilarityModel(nn.Module):
    """
    The complete Siamese network.
    """
    def __init__(self, config_path, checkpoint_path, embedding_dim=512):
        super().__init__()
        # A single instance of the encoder is created and shared.
        self.encoder = MassFormerEncoder(config_path, checkpoint_path)
        self.similarity_head = SimilarityHead(embedding_dim)

    def forward(self, batch_A, batch_B):
        # The SAME encoder instance is called on both inputs.
        embedding_A = self.encoder(batch_A)
        embedding_B = self.encoder(batch_B)
        
        # Combine embeddings by taking the absolute difference.
        # This forces the model to learn a distance metric.
        combined_embedding = torch.abs(embedding_A - embedding_B)
        
        # Pass the combined embedding to the similarity head
        predicted_similarity = self.similarity_head(combined_embedding)
        
        return predicted_similarity