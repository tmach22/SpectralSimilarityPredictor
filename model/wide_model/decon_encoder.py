import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import the base MassFormerEncoder from your Phase 1 wide model script
from classifier_siamesemodel_wide import MassFormerEncoder 

class StageOneMetricModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str):
        super().__init__()
        
        # Initialize and load the MassSpec fine-tuned weights using the merged config
        # The reinit_num_pt_layers: -1 flag in the config handles the transformer reset natively
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        
        print("\n" + "="*50)
        print("STAGE 1: ASYMMETRIC METRIC LEARNING")
        print(" -> Loading Massformer weights with native YAML configuration.")
        print(" -> Proceeding to fine-tune with Order Embeddings Loss.")
        print("="*50 + "\n")

        # 2. EXPLICIT POST-LOAD RESET (Curing the Global Topology Bias)
        print("\n" + "="*50)
        print("STAGE 1: ASYMMETRIC METRIC LEARNING")
        print(" -> Applying Transfer & Reset Strategy (Post-Checkpoint Load)")
        
        # Dig into the nested model architecture to find the Graphormer
        gf_embedder = self.encoder.encoder
        graphormer_encoder = gf_embedder.encoder
        num_layers = len(graphormer_encoder.graph_encoder.layers)
        
        # Forcefully wipe the transformer layers and LayerNorms, but keep the atom embeddings
        graphormer_encoder.reinit_encoder_layer_parameters(num_layers, reinit_layernorm=True)
        
        print(f" -> ✓ Preserved pre-trained atomic & structural token embeddings.")
        print(f" -> ✓ Randomly re-initialized all {num_layers} Transformer Attention Layers.")
        print("="*50 + "\n")
        
    def forward(self, molecule_A_data, molecule_B_data):
        # Extract the 768-D Readout Nodes
        emb_A = self.encoder({'gf_v2_data': molecule_A_data})
        emb_B = self.encoder({'gf_v2_data': molecule_B_data})
        
        # Enforce positive orthant for Order Embeddings constraint (h_light <= h_heavy)
        emb_A = F.relu(emb_A)
        emb_B = F.relu(emb_B)
        
        return emb_A, emb_B

class OrderEmbeddingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_A, emb_B, mass_A, mass_B, labels):
        # 1. Determine Heavy vs Light dynamically based on physical mass
        # is_A_heavy shape: [Batch, 1]
        is_A_heavy = (mass_A >= mass_B)
        
        h_heavy = torch.where(is_A_heavy, emb_A, emb_B)
        h_light = torch.where(is_A_heavy, emb_B, emb_A)

        # 2. Asymmetric Violation (Positive Pairs)
        # We penalize dimensions where the light molecule's feature > heavy molecule's feature
        violation = F.relu(h_light - h_heavy)
        positive_loss = torch.norm(violation, p=2, dim=1)**2
        
        # 3. Contrastive Distance (Negative Pairs)
        # Standard margin-based push for completely unrelated graphs
        euclidean_dist = torch.norm(emb_A - emb_B, p=2, dim=1)
        negative_loss = F.relu(self.margin - euclidean_dist)**2
        
        # 4. Apply labels (1 = match, 0 = mismatch)
        loss = (labels * positive_loss) + ((1.0 - labels) * negative_loss)
        
        return loss.mean()