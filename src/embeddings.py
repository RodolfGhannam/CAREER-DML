"""
CAREER-DML: Career Embeddings Module - v3.4
Camada 2 - Four embedding variants for causal inference with career sequences.

Variants:
    1. PredictiveGRU: Outcome-only baseline (no causal adjustment).
    2. CausalGRU (VIB): Veitch et al. (2020) with Variational Information Bottleneck.
    3. DebiasedGRU (Adversarial): Adversarial debiasing to remove treatment signal.
    4. TwoStageCausalGRU: Veitch-faithful two-stage approach (pre-train on Y, fine-tune with VIB+T).

References:
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
        """Reparameterization trick for VIB."""
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

    Uses gradient reversal to produce embeddings informative about Y but not T.

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


# =============================================================================
# VARIANTE 4: Two-Stage Causal GRU — Veitch-Faithful Implementation
# =============================================================================

class TwoStageCausalGRU(nn.Module):
    """Two-stage GRU implementing Veitch et al. (2020) faithfully.

    Stage 1: Pre-train GRU encoder on outcome prediction (like PredictiveGRU).
    Stage 2: Freeze encoder, add VIB compression + dual heads (Y, T).

    This mirrors the original Veitch approach of using pre-trained embeddings
    (BERT in their case) followed by supervised dimensionality reduction,
    rather than training the encoder and VIB jointly from scratch.

    Architecture:
        Stage 1: Embedding → GRU → Linear → Y (pre-training)
        Stage 2: [frozen GRU] → VIB(mu, logvar) → phi
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
        # Shared encoder (frozen after Stage 1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_pretrain = nn.Linear(hidden_dim, 1)  # Stage 1 head

        # VIB compression (Stage 2 only)
        self.fc_mu = nn.Linear(hidden_dim, phi_dim)
        self.fc_logvar = nn.Linear(hidden_dim, phi_dim)

        # Dual prediction heads (Stage 2 only)
        self.fc_y = nn.Linear(phi_dim, 1)
        self.fc_t = nn.Linear(phi_dim, 1)

        self.phi_dim = phi_dim
        self.hidden_dim = hidden_dim
        self._stage = 1  # Track current training stage

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: sequences → hidden state."""
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        return h_n.squeeze(0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VIB."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1 forward: predict Y from raw hidden state."""
        h = self._encode(x)
        return self.fc_pretrain(h).squeeze(-1)

    def forward_stage2(self, x: torch.Tensor) -> tuple:
        """Stage 2 forward: VIB compression + dual heads."""
        h = self._encode(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        phi = self.reparameterize(mu, logvar)
        y_pred = self.fc_y(phi).squeeze(-1)
        t_pred = self.fc_t(phi).squeeze(-1)
        return y_pred, t_pred, mu, logvar

    def forward(self, x: torch.Tensor):
        """Route to the appropriate stage."""
        if self._stage == 1:
            return self.forward_stage1(x)
        return self.forward_stage2(x)

    def freeze_encoder(self):
        """Freeze the GRU encoder for Stage 2 fine-tuning."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.gru.parameters():
            param.requires_grad = False
        for param in self.fc_pretrain.parameters():
            param.requires_grad = False
        self._stage = 2

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the VIB mean as the career embedding (deterministic)."""
        h = self._encode(x)
        return self.fc_mu(h)


def train_twostage_causal_embedding(
    model: TwoStageCausalGRU,
    dataloader,
    epochs_stage1: int = 10,
    epochs_stage2: int = 10,
    lr: float = 1e-3,
    beta_vib: float = 0.01,
    alpha_t: float = 0.5,
) -> dict:
    """Train the Two-Stage Causal GRU following Veitch et al. (2020).

    Stage 1: Pre-train the GRU encoder to predict Y (outcome).
             This learns good sequential representations before compression.
    Stage 2: Freeze encoder. Train VIB + dual heads (Y, T).
             This applies causal compression to pre-learned representations.

    Returns:
        dict with 'stage1_losses' and 'stage2_losses'.
    """
    results = {'stage1_losses': [], 'stage2_losses': []}

    # --- Stage 1: Pre-train encoder on outcome prediction ---
    print("  [Two-Stage] Stage 1: Pre-training encoder on outcome prediction...")
    model._stage = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs_stage1):
        epoch_loss = 0.0
        n_batches = 0
        for sequences, treatments, outcomes in dataloader:
            optimizer.zero_grad()
            y_pred = model.forward_stage1(sequences)
            loss = F.mse_loss(y_pred, outcomes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        results['stage1_losses'].append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Stage 1 Epoch {epoch+1}/{epochs_stage1} — Loss: {avg_loss:.4f}")

    # --- Stage 2: Freeze encoder, train VIB + dual heads ---
    print("  [Two-Stage] Stage 2: Freezing encoder, training VIB + dual heads...")
    model.freeze_encoder()

    # Only optimise Stage 2 parameters
    stage2_params = list(model.fc_mu.parameters()) + \
                    list(model.fc_logvar.parameters()) + \
                    list(model.fc_y.parameters()) + \
                    list(model.fc_t.parameters())
    optimizer2 = torch.optim.Adam(stage2_params, lr=lr)

    model.train()
    for epoch in range(epochs_stage2):
        epoch_loss = 0.0
        n_batches = 0
        for sequences, treatments, outcomes in dataloader:
            optimizer2.zero_grad()
            y_pred, t_pred, mu, logvar = model.forward_stage2(sequences)

            loss_y = F.mse_loss(y_pred, outcomes)
            loss_t = F.binary_cross_entropy_with_logits(t_pred, treatments.float())
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = loss_y + alpha_t * loss_t + beta_vib * kl_div
            loss.backward()
            optimizer2.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        results['stage2_losses'].append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Stage 2 Epoch {epoch+1}/{epochs_stage2} — Loss: {avg_loss:.4f}")

    return results
