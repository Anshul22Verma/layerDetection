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


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, p1, z1, p2, z2):
        # Cosine similarity loss (negative cosine similarity between projections and predictions)
        # This assumes the input is normalized before computing similarity
        loss = -0.5 * (F.cosine_similarity(p1, z2.detach(), dim=-1) + F.cosine_similarity(p2, z1.detach(), dim=-1))
        return loss.mean()


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()
    
    def forward(self, z1, z2, target_z1, target_z2,):
        # Normalize the embeddings and predictions
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        target_z1 = F.normalize(target_z1, dim=-1, p=2)
        target_z2 = F.normalize(target_z2, dim=-1, p=2)

        # Cosine similarity between z1 and target_z2, and z2 and target_z1
        sim_q_k_1 = torch.sum(z1 * target_z2, dim=-1)
        sim_q_k_2 = torch.sum(z2 * target_z1, dim=-1)
        
        # Loss: mean of the cosine similarities
        loss = - (sim_q_k_1 + sim_q_k_2) / 2.0
        return loss.mean()
