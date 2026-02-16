"""
MURD (Multi-site Unsupervised Representation Disentangler) - 2.5D Original Implementation in PyTorch
for Multi-Site Medical Image Harmonization

Based on:
Liu, S. & Yap, P.T. (2024). Learning multi-site harmonization of magnetic resonance 
images without traveling human phantoms. Communications Engineering.
"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Tuple

from logger import LossLogger
import time

from model_2_5D_components import *


# ============================================================================
# COMPLETE MURD MODEL
# ============================================================================

class MURD_2_5D(nn.Module):
    """Complete MURD model for 2.5D slices"""
    def __init__(self, 
                 in_channels: int = 3,  # 2.5D: 3 adjacent slices
                 num_sites: int = 3,
                 style_dim: int = 64,
                 latent_dim: int = 16,
                 base_channels: int = 64):
        super().__init__()
        
        self.num_sites = num_sites
        self.style_dim = style_dim
        self.latent_dim = latent_dim
        
        self.content_encoder = ContentEncoder2D(in_channels, base_channels)
        self.style_encoder = StyleEncoder2D(in_channels, num_sites, style_dim, base_channels // 2)
        self.generator = Generator2D(base_channels * 4, style_dim, in_channels)
        self.style_generator = StyleGenerator2D(latent_dim, 256, style_dim, num_sites)
        self.discriminator = Discriminator2D(in_channels, num_sites, base_channels)
    
    def encode_content(self, x):
        return self.content_encoder(x)
    
    def encode_style(self, x, site_idx):
        return self.style_encoder(x, site_idx)
    
    def generate_style(self, batch_size, site_idx, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.style_generator(z, site_idx)
    
    def decode(self, content, style):
        return self.generator(content, style)
    
    def forward_site_specific(self, x_i, site_i, site_j, device):
        """Forward pass for site-specific harmonization"""
        # Encode content and style from source image
        c_i = self.encode_content(x_i)
        s_i = self.encode_style(x_i, site_i)
        
        # Generate target style
        s_j = self.generate_style(x_i.size(0), site_j, device)
        
        # Generate harmonized image
        x_j_hat = self.decode(c_i, s_j)
        
        # Encode harmonized image
        c_j_tilde = self.encode_content(x_j_hat)
        s_j_tilde = self.encode_style(x_j_hat, site_j)
        
        # Reconstruct original image (identity)
        x_i_recon = self.decode(c_i, s_i)
        
        return {
            'c_i': c_i,
            's_i': s_i,
            's_j': s_j,
            'x_j_hat': x_j_hat,
            'c_j_tilde': c_j_tilde,
            's_j_tilde': s_j_tilde,
            'x_i_recon': x_i_recon
        }




# ============================================================================
# LOSS FUNCTIONS (Memory-efficient)
# ============================================================================

class MURDLoss:
    """Loss functions with correct weights from paper"""
    def __init__(self):
        # From paper: λadv = 1, λcont = 10, λca = 0.01, λsd = 1, λsty = 10, λcyc = 10, λg = 0.1, λid = 10
        self.lambda_adv = 1.0
        self.lambda_cont = 10.0
        self.lambda_sty = 10.0
        self.lambda_cyc = 10.0
        self.lambda_id = 10.0
        self.lambda_ca = 0.01
        self.lambda_sd = 1.0
        self.lambda_g = 0.1
    
    def adversarial_loss(self, pred, target_is_real):
        if target_is_real:
            return F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
        else:
            return F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred))
    
    def content_consistency_loss(self, c_i, c_j_tilde):
        # Ensures content encodings are similar before and after harmonization
        return F.l1_loss(c_i, c_j_tilde)
    
    def style_consistency_loss(self, s_j, s_j_tilde):
        # Ensures style encodings are similar before and after harmonization
        return F.l1_loss(s_j, s_j_tilde)
    
    def identity_loss(self, x_i, x_i_recon):
        # Ensures consistency between original and reconstructed image (identity mapping)
        return F.l1_loss(x_i, x_i_recon)
    
    def content_alignment_loss(self, c_i):
        # KL divergence to enforce N(0, I) distribution
        mean = c_i.mean(dim=[2, 3])
        var = c_i.var(dim=[2, 3])
        kl_div = 0.5 * (mean.pow(2) + var - var.log() - 1)
        return kl_div.mean()
    
    def style_diversity_loss(self, x_j_hat1, x_j_hat2):
        # Encourages diversity in generated styles by penalizing similarity between two different style generations
        return -F.l1_loss(x_j_hat1, x_j_hat2)
    
    def gradient_loss(self, x, x_hat):
        def image_gradients(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy
        
        dx, dy = image_gradients(x)
        dx_hat, dy_hat = image_gradients(x_hat)
        
        return F.l1_loss(dx, dx_hat) + F.l1_loss(dy, dy_hat)
    
    def compute_generator_loss(self, outputs, x_i, discriminator, site_j):
        losses = {}
        
        # Adversarial loss
        d_fake = discriminator(outputs['x_j_hat'], site_j)
        losses['adv'] = self.lambda_adv * self.adversarial_loss(d_fake, True)
        
        # Content consistency
        losses['cont'] = self.lambda_cont * self.content_consistency_loss(
            outputs['c_i'], outputs['c_j_tilde']
        )
        
        # Style consistency
        losses['sty'] = self.lambda_sty * self.style_consistency_loss(
            outputs['s_j'], outputs['s_j_tilde']
        )
        
        # Identity loss
        losses['id'] = self.lambda_id * self.identity_loss(x_i, outputs['x_i_recon'])
        
        # Content alignment
        losses['ca'] = self.lambda_ca * self.content_alignment_loss(outputs['c_i'])
        
        # Cycle consistency with gradient loss
        # Note: We use identity as proxy for cycle (memory efficient)
        losses['cyc'] = self.lambda_cyc * (
            F.l1_loss(x_i, outputs['x_i_recon']) +
            self.lambda_g * self.gradient_loss(x_i, outputs['x_i_recon'])
        )
        
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_discriminator_loss(self, x_real, x_fake, discriminator, site_idx):
        d_real = discriminator(x_real, site_idx)
        d_fake = discriminator(x_fake.detach(), site_idx)
        
        loss_real = self.adversarial_loss(d_real, True)
        loss_fake = self.adversarial_loss(d_fake, False)
        
        return loss_real + loss_fake

# ============================================================================
# TRAINING CLASS
# ============================================================================

class MURDTrainer:
    """Trainer with mixed precision and gradient accumulation"""
    def __init__(self, 
                 experiment_name: str,
                 model: MURD_2_5D,
                 epochs:int = 100,
                 num_sites: int = 2,
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.5, 0.999),
                 device: str = 'cuda',
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 4):
        
        self.experiment_name = experiment_name
        self.model = model.to(device)
        self.num_sites = num_sites
        self.epochs = epochs
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.criterion = MURDLoss()
        
        self.opt_gen = Adam(
            list(model.content_encoder.parameters()) +
            list(model.style_encoder.parameters()) +
            list(model.generator.parameters()) +
            list(model.style_generator.parameters()),
            lr=lr, betas=betas
        )
        
        self.opt_disc = Adam(
            model.discriminator.parameters(),
            lr=lr, betas=betas
        )
        
        self.logger = LossLogger(log_dir='logs/', experiment_name=experiment_name)
        
        # Save configuration
        config = {
            'num_sites': num_sites,
            'epochs': epochs,
            'lr': lr,
            'betas': betas,
            'use_amp': use_amp,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'device': device,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'loss_weights': {
                'lambda_adv': self.criterion.lambda_adv,
                'lambda_cont': self.criterion.lambda_cont,
                'lambda_sty': self.criterion.lambda_sty,
                'lambda_cyc': self.criterion.lambda_cyc,
                'lambda_id': self.criterion.lambda_id,
                'lambda_ca': self.criterion.lambda_ca,
                'lambda_sd': self.criterion.lambda_sd,
            }
        }
        self.logger.save_config(config)

        # Mixed precision training
        self.scaler_gen = GradScaler(enabled=use_amp)
        self.scaler_disc = GradScaler(enabled=use_amp)
    
    def train_step(self, batch: Dict[int, torch.Tensor], accumulation_step: int):
        """Single training step with gradient accumulation"""
        losses = {}
        
        sites = list(batch.keys())
        if len(sites) < 2:
            return None
        
        site_i, site_j = np.random.choice(sites, 2, replace=False)
        
        x_i = batch[site_i].to(self.device)
        x_j = batch[site_j].to(self.device)
        
        # Train Generator
        with autocast():
            outputs = self.model.forward_site_specific(x_i, site_i, site_j, self.device)
            gen_losses = self.criterion.compute_generator_loss(
                outputs, x_i, self.model.discriminator, site_j
            )
        
        self.opt_gen.zero_grad()
        self.scaler_gen.scale(gen_losses['total']).backward()
        self.scaler_gen.step(self.opt_gen)
        self.scaler_gen.update()
        
        # Train Discriminator
        with autocast():
            disc_loss = self.criterion.compute_discriminator_loss(
                x_j, outputs['x_j_hat'], self.model.discriminator, site_j
            )
        
        self.opt_disc.zero_grad()
        self.scaler_disc.scale(disc_loss).backward()
        self.scaler_disc.step(self.opt_disc)
        self.scaler_disc.update()
        
        losses = {f'gen_{k}': v.item() for k, v in gen_losses.items()}
        losses['disc'] = disc_loss.item()
        
        del outputs
        torch.cuda.empty_cache()
        
        return losses
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch, batch_idx)
            
            if losses is None:
                continue
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: " +
                      ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))
        
        # Calculate epoch averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        epoch_time = time.time() - start_time
        
        # Get current learning rate
        current_lr = self.opt_gen.param_groups[0]['lr']
        
        # Log epoch summary
        self.logger.log_epoch(epoch, avg_losses, epoch_time, current_lr)
        return avg_losses
    
    def save_checkpoint(self, path, epoch, **kwargs):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            **kwargs
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        return checkpoint



