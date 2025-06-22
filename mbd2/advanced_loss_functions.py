#!/usr/bin/env python3
"""
Advanced Loss Functions and Training Optimizations
- Enhanced InfoNCE loss
- Adaptive temperature
- Focal contrastive loss
- Multi-scale contrastive learning
- Advanced learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AdaptiveTemperatureContrastiveLoss(nn.Module):
    """
    Contrastive loss with learnable adaptive temperature
    """
    
    def __init__(self, initial_temperature=0.07, min_temp=0.01, max_temp=1.0, 
                 learn_temperature=True):
        super().__init__()
        
        self.min_temp = min_temp
        self.max_temp = max_temp
        
        if learn_temperature:
            # Learnable temperature parameter
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(initial_temperature)))
        
    @property
    def temperature(self):
        # Constrain temperature to reasonable range
        temp = torch.exp(self.log_temperature)
        return torch.clamp(temp, self.min_temp, self.max_temp)
    
    def forward(self, eeg_embeddings, clip_embeddings):
        """
        Compute adaptive temperature contrastive loss
        
        Args:
            eeg_embeddings: [batch, embedding_dim]
            clip_embeddings: [batch, embedding_dim]
            
        Returns:
            loss: Contrastive loss
            accuracy: Top-1 accuracy
            temperature: Current temperature value
        """
        batch_size = eeg_embeddings.size(0)
        
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=1)
        clip_embeddings = F.normalize(clip_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(eeg_embeddings, clip_embeddings.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=eeg_embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(similarity_matrix, dim=1)
            accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy, self.temperature.item()

class FocalContrastiveLoss(nn.Module):
    """
    Focal contrastive loss to focus on hard negatives
    """
    
    def __init__(self, temperature=0.07, alpha=1.0, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, eeg_embeddings, clip_embeddings):
        """
        Compute focal contrastive loss
        
        Args:
            eeg_embeddings: [batch, embedding_dim]
            clip_embeddings: [batch, embedding_dim]
            
        Returns:
            loss: Focal contrastive loss
            accuracy: Top-1 accuracy
        """
        batch_size = eeg_embeddings.size(0)
        
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=1)
        clip_embeddings = F.normalize(clip_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(eeg_embeddings, clip_embeddings.T) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=eeg_embeddings.device)
        
        # Compute probabilities
        log_probs = F.log_softmax(similarity_matrix, dim=1)
        probs = F.softmax(similarity_matrix, dim=1)
        
        # Get probabilities for correct class
        correct_probs = probs[torch.arange(batch_size), labels]
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - correct_probs) ** self.gamma
        
        # Compute focal loss
        focal_loss = -focal_weight * log_probs[torch.arange(batch_size), labels]
        loss = focal_loss.mean()
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(similarity_matrix, dim=1)
            accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy

class MultiScaleContrastiveLoss(nn.Module):
    """
    Multi-scale contrastive learning with different embedding dimensions
    """
    
    def __init__(self, embedding_dims=[256, 512, 1024], temperature=0.07, weights=None):
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.temperature = temperature
        
        if weights is None:
            self.weights = [1.0] * len(embedding_dims)
        else:
            self.weights = weights
        
        # Projection heads for different scales
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, dim),  # Assuming input is 512-dim
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ) for dim in embedding_dims
        ])
        
    def forward(self, eeg_embeddings, clip_embeddings):
        """
        Compute multi-scale contrastive loss
        
        Args:
            eeg_embeddings: [batch, 512]
            clip_embeddings: [batch, 512]
            
        Returns:
            total_loss: Weighted sum of losses at different scales
            accuracy: Average accuracy across scales
        """
        total_loss = 0
        total_accuracy = 0
        batch_size = eeg_embeddings.size(0)
        
        for i, (proj_head, weight) in enumerate(zip(self.projection_heads, self.weights)):
            # Project to different embedding dimensions
            eeg_proj = proj_head(eeg_embeddings)
            clip_proj = proj_head(clip_embeddings)
            
            # Normalize
            eeg_proj = F.normalize(eeg_proj, p=2, dim=1)
            clip_proj = F.normalize(clip_proj, p=2, dim=1)
            
            # Compute similarity
            similarity = torch.matmul(eeg_proj, clip_proj.T) / self.temperature
            
            # Labels
            labels = torch.arange(batch_size, device=eeg_embeddings.device)
            
            # Loss
            loss = F.cross_entropy(similarity, labels)
            total_loss += weight * loss
            
            # Accuracy
            with torch.no_grad():
                predictions = torch.argmax(similarity, dim=1)
                accuracy = (predictions == labels).float().mean()
                total_accuracy += accuracy
        
        avg_accuracy = total_accuracy / len(self.embedding_dims)
        
        return total_loss, avg_accuracy

class AdvancedInfoNCELoss(nn.Module):
    """
    Advanced InfoNCE loss with hard negative mining and momentum
    """
    
    def __init__(self, temperature=0.07, momentum=0.999, queue_size=4096, 
                 hard_negative_ratio=0.3):
        super().__init__()
        
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size
        self.hard_negative_ratio = hard_negative_ratio
        
        # Memory bank for negative samples
        self.register_buffer("queue", torch.randn(queue_size, 512))
        self.queue = F.normalize(self.queue, p=2, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update memory bank with new keys
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, eeg_embeddings, clip_embeddings):
        """
        Compute advanced InfoNCE loss
        
        Args:
            eeg_embeddings: [batch, embedding_dim]
            clip_embeddings: [batch, embedding_dim]
            
        Returns:
            loss: InfoNCE loss with hard negatives
            accuracy: Top-1 accuracy
        """
        batch_size = eeg_embeddings.size(0)
        
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=1)
        clip_embeddings = F.normalize(clip_embeddings, p=2, dim=1)
        
        # Positive pairs (within batch)
        pos_similarities = torch.sum(eeg_embeddings * clip_embeddings, dim=1, keepdim=True)
        
        # Negative pairs (from memory bank)
        neg_similarities = torch.matmul(eeg_embeddings, self.queue.T)
        
        # Hard negative mining
        if self.hard_negative_ratio > 0:
            num_hard_negatives = int(self.queue_size * self.hard_negative_ratio)
            
            # Select hardest negatives (highest similarity)
            hard_neg_indices = torch.topk(neg_similarities, num_hard_negatives, dim=1)[1]
            hard_neg_similarities = torch.gather(neg_similarities, 1, hard_neg_indices)
            
            # Combine with random negatives
            num_random_negatives = self.queue_size - num_hard_negatives
            random_indices = torch.randint(0, self.queue_size, 
                                         (batch_size, num_random_negatives),
                                         device=eeg_embeddings.device)
            random_neg_similarities = torch.gather(neg_similarities, 1, random_indices)
            
            neg_similarities = torch.cat([hard_neg_similarities, random_neg_similarities], dim=1)
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_similarities, neg_similarities], dim=1) / self.temperature
        
        # Labels (positive pairs are at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=eeg_embeddings.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Update memory bank
        self._dequeue_and_enqueue(clip_embeddings)
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy

class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        """
        Update learning rate based on current epoch
        """
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def test_loss_functions():
    """
    Test advanced loss functions
    """
    print("🧪 Testing Advanced Loss Functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    embedding_dim = 512
    
    # Create test embeddings
    eeg_embeddings = torch.randn(batch_size, embedding_dim).to(device)
    clip_embeddings = torch.randn(batch_size, embedding_dim).to(device)
    
    print(f"📊 Test embeddings shape: {eeg_embeddings.shape}")
    
    # Test adaptive temperature loss
    print("\n🔧 Testing Adaptive Temperature Loss:")
    adaptive_loss = AdaptiveTemperatureContrastiveLoss(learn_temperature=True).to(device)
    loss, acc, temp = adaptive_loss(eeg_embeddings, clip_embeddings)
    print(f"   Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}, Temperature: {temp:.4f}")
    
    # Test focal contrastive loss
    print("\n🔧 Testing Focal Contrastive Loss:")
    focal_loss = FocalContrastiveLoss(gamma=2.0).to(device)
    loss, acc = focal_loss(eeg_embeddings, clip_embeddings)
    print(f"   Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}")
    
    # Test multi-scale loss
    print("\n🔧 Testing Multi-Scale Loss:")
    multiscale_loss = MultiScaleContrastiveLoss(embedding_dims=[256, 512, 1024]).to(device)
    loss, acc = multiscale_loss(eeg_embeddings, clip_embeddings)
    print(f"   Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}")
    
    # Test advanced InfoNCE
    print("\n🔧 Testing Advanced InfoNCE Loss:")
    infonce_loss = AdvancedInfoNCELoss(queue_size=1024).to(device)
    loss, acc = infonce_loss(eeg_embeddings, clip_embeddings)
    print(f"   Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}")
    
    # Test learning rate scheduler
    print("\n🔧 Testing Warmup Cosine Scheduler:")
    optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, max_epochs=100, base_lr=1e-4)
    
    lrs = []
    for epoch in range(20):
        lr = scheduler.step(epoch)
        lrs.append(lr)
    
    print(f"   Epoch 0 LR: {lrs[0]:.2e}")
    print(f"   Epoch 10 LR: {lrs[10]:.2e}")
    print(f"   Epoch 19 LR: {lrs[19]:.2e}")
    
    print("✅ Advanced loss functions test completed!")

if __name__ == "__main__":
    test_loss_functions()
