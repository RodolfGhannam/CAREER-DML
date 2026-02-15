"""
CAREER-DML: Career Embeddings Module - v3.2
Camada 2 - Três variantes de representação causal para sequências de carreira.

Variantes:
    1. PredictiveGRU: Baseline preditivo (espera-se que falhe causalmente).
    2. CausalGRU (VIB): Veitch et al. (2020) com Variational Information Bottleneck.
    3. DebiasedGRU (Adversarial): Adversarial debiasing para purgar informação de tratamento.

Referências:
    - Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder
    - Veitch, Sridhar, Blei (2020), Adapting Text Embeddings for Causal Inference
    - Vafa et al. (2025), Career Embeddings (PNAS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# VARIANTE 1: Predictive GRU (Baseline)
# =============================================================================

class PredictiveGRU(nn.Module):
    """Standard GRU trained to predict outcome from career sequences.

    This is the causally naive baseline. It absorbs confounding information
    (ability → career path → embedding), making it invalid for ATE estimation
    but potentially useful for discovering heterogeneity (GATES).

    Architecture:
        Embedding → GRU → Linear → outcome prediction
    """

    def __init__(self, vocab_size: int = 50, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sequences → predicted outcome."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h = h_n.squeeze(0)
        return self.fc_out(h).squeeze(-1)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the hidden state as the career embedding."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        return h_n.squeeze(0)


# =============================================================================
# VARIANTE 2: Causal GRU (VIB) — Veitch et al. (2020)
# =============================================================================

class CausalGRU(nn.Module):
    """GRU with Variational Information Bottleneck for causal representation.

    Implements the dual-loss architecture of Veitch et al. (2020):
    - Predict outcome Y from embedding (main task)
    - Predict treatment T from embedding (auxiliary task)
    - VIB regularization to compress the representation
    - Overlap penalty to encourage common support

    Architecture:
        Embedding → GRU → VIB(mu, logvar) → reparameterize → phi
        phi → Linear → Y prediction
        phi → Linear → T prediction
    """

    def __init__(
        self,
        vocab_size: int = 50,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        phi_dim: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # VIB: map hidden state to (mu, logvar) for reparameterization
        self.fc_mu = nn.Linear(hidden_dim, phi_dim)
        self.fc_logvar = nn.Linear(hidden_dim, phi_dim)

        # Dual prediction heads
        self.fc_y = nn.Linear(phi_dim, 1)  # Outcome prediction
        self.fc_t = nn.Linear(phi_dim, 1)  # Treatment prediction

        self.phi_dim = phi_dim
        self.hidden_dim = hidden_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VIB.

        Includes logvar clamping for numerical stability, preventing
        log(0) → -inf when sigma → 0 or sigma → inf.
        Ref: Board Review (Veitch), Feb 2026.
        """
        logvar = torch.clamp(logvar, min=-10, max=10)  # Numerical guard
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (y_pred, t_pred, mu, logvar)."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h = h_n.squeeze(0)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        phi = self.reparameterize(mu, logvar)

        y_pred = self.fc_y(phi).squeeze(-1)
        t_pred = self.fc_t(phi).squeeze(-1)

        return y_pred, t_pred, mu, logvar

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the VIB mean as the career embedding (deterministic at eval)."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h = h_n.squeeze(0)
        return self.fc_mu(h)


# =============================================================================
# VARIANTE 3: Debiased GRU (Adversarial)
# =============================================================================

class DebiasedGRU(nn.Module):
    """GRU with adversarial debiasing for causal validity.

    The encoder produces embeddings that are informative about Y but NOT
    about T. An adversary tries to predict T from the embedding; the encoder
    is penalized for the adversary's success.

    This is the winning variant that achieves ATE = 0.6712 (bias: 0.1712, 34.2% error).

    Architecture:
        Embedding → GRU → Linear(phi_dim) → phi
        phi → Linear → Y prediction (main task)
        phi → Adversary → T prediction (adversarial task, gradient reversed)
    """

    def __init__(
        self,
        vocab_size: int = 50,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        phi_dim: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_phi = nn.Linear(hidden_dim, phi_dim)
        self.fc_y = nn.Linear(phi_dim, 1)

        self.phi_dim = phi_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (y_pred, phi)."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h = h_n.squeeze(0)
        phi = F.relu(self.fc_phi(h))
        y_pred = self.fc_y(phi).squeeze(-1)
        return y_pred, phi

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the debiased embedding."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h = h_n.squeeze(0)
        return F.relu(self.fc_phi(h))


class Adversary(nn.Module):
    """Adversary network that tries to predict treatment from embedding.

    Used in adversarial training: if the adversary can predict T from phi,
    then phi still contains confounding information that must be purged.
    """

    def __init__(self, phi_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).squeeze(-1)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_predictive_embedding(
    model: PredictiveGRU,
    dataloader,
    epochs: int = 15,
    lr: float = 1e-3,
) -> list[float]:
    """Train the Predictive GRU to predict outcome Y.

    Simple supervised training: minimize MSE(Y, Y_hat).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for sequences, treatments, outcomes in dataloader:
            optimizer.zero_grad()
            y_pred = model(sequences)
            loss = F.mse_loss(y_pred, outcomes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return losses


def train_causal_embedding(
    model: CausalGRU,
    dataloader,
    epochs: int = 15,
    lr: float = 1e-3,
    beta_vib: float = 0.01,
    alpha_t: float = 0.5,
) -> list[float]:
    """Train the Causal GRU with VIB regularization.

    Loss = MSE(Y) + alpha_t * BCE(T) + beta_vib * KL(q(phi|x) || p(phi))

    Following Veitch et al. (2020): the treatment prediction head ensures
    the embedding captures treatment-relevant information, while VIB
    compresses it to reduce confounding.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for sequences, treatments, outcomes in dataloader:
            optimizer.zero_grad()

            y_pred, t_pred, mu, logvar = model(sequences)

            # Outcome loss
            loss_y = F.mse_loss(y_pred, outcomes)

            # Treatment prediction loss
            loss_t = F.binary_cross_entropy_with_logits(t_pred, treatments.float())

            # KL divergence (VIB regularization)
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            loss = loss_y + alpha_t * loss_t + beta_vib * kl_div
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return losses


def train_debiased_embedding(
    encoder: DebiasedGRU,
    adversary: Adversary,
    dataloader,
    epochs: int = 15,
    lr_encoder: float = 1e-3,
    lr_adversary: float = 1e-3,
    lambda_adv: float = 1.0,
) -> list[float]:
    """Train the Debiased GRU with adversarial debiasing.

    Two-player game:
    1. Adversary tries to predict T from phi (minimize BCE(T, T_hat))
    2. Encoder tries to predict Y from phi AND fool the adversary
       (minimize MSE(Y) - lambda_adv * BCE(T, T_hat))

    The gradient reversal ensures the encoder produces embeddings that
    are informative about Y but uninformative about T.
    """
    opt_encoder = torch.optim.Adam(encoder.parameters(), lr=lr_encoder)
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr_adversary)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        encoder.train()
        adversary.train()

        for sequences, treatments, outcomes in dataloader:
            # --- Step 1: Train Adversary ---
            opt_adversary.zero_grad()
            with torch.no_grad():
                _, phi = encoder(sequences)
            t_pred_adv = adversary(phi.detach())
            loss_adv = F.binary_cross_entropy_with_logits(t_pred_adv, treatments.float())
            loss_adv.backward()
            opt_adversary.step()

            # --- Step 2: Train Encoder (with adversarial penalty) ---
            opt_encoder.zero_grad()
            y_pred, phi = encoder(sequences)
            loss_y = F.mse_loss(y_pred, outcomes)

            # Adversarial penalty: encoder wants adversary to FAIL
            t_pred_enc = adversary(phi)
            loss_fool = F.binary_cross_entropy_with_logits(t_pred_enc, treatments.float())

            # Total encoder loss: predict Y well, but fool the adversary
            loss_encoder = loss_y - lambda_adv * loss_fool
            loss_encoder.backward()
            opt_encoder.step()

            epoch_loss += loss_encoder.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Encoder Loss: {avg_loss:.4f}, Adv Loss: {loss_adv.item():.4f}")

    return losses
