import torch
import troch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """Compute NT-Xent loss for contrastive learning."""
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # Concatenate positive pairs
        z = F.normalize(z, dim=1)  # Normalize embeddings

        similarity = torch.matmul(z, z.T)  # Cosine similarity --> To find out similarity between the positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        similarity = similarity[~mask].view(2 * batch_size, -1)

        positives = torch.exp(torch.sum(z1 * z2, dim=1) / self.temperature)
        negatives = torch.exp(similarity / self.temperature).sum(dim=1)

        loss = -torch.log(positives / negatives).mean()
        return loss

class MoCoLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, memory_size=4096):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature
        self.memory_size = memory_size
        self.register_buffer('queue', torch.randn(memory_size, 128))  # Assuming embedding dimension is 128
        self.queue = F.normalize(self.queue, dim=1)  # Normalize the queue to unit vectors

    def forward(self, query, key, update_queue=True):
        """
        Forward pass for the MoCo loss. The `query` and `key` are the outputs of the query encoder and key encoder.
        
        Arguments:
        - query (Tensor): Query feature with shape (batch_size, embedding_dim)
        - key (Tensor): Key feature with shape (batch_size, embedding_dim)
        - update_queue (bool): Whether to update the memory queue with new keys.
        
        Returns:
        - loss (Tensor): The contrastive loss.
        """
        batch_size = query.shape[0]
        
        # Normalize query and key embeddings
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        # Compute similarity between query and key
        sim_qk = torch.matmul(query, key.T) / self.temperature
        
        # Similarity with the queue (negative samples)
        sim_qk_neg = torch.matmul(query, self.queue.T) / self.temperature
        
        # Compute loss for each query
        logits = torch.cat([sim_qk.unsqueeze(1), sim_qk_neg], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        if update_queue:
            # Update the memory queue with new keys
            with torch.no_grad():
                # Use the key as the key feature for the momentum encoder
                self.queue = torch.cat([self.queue[batch_size:], key], dim=0)
                self.queue = F.normalize(self.queue, dim=1)  # Normalize after update

        return loss

