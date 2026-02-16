"""
CAREER-DML: Utility Functions
Common functions shared across pipeline scripts.

Extracted from main.py to eliminate code duplication.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def pad_sequences(sequences: list[list[int]], max_len: int = 10) -> torch.Tensor:
    """Pad sequences with zeros to uniform length.

    Args:
        sequences: List of integer sequences (occupation IDs).
        max_len: Maximum sequence length (shorter sequences are padded,
                 longer sequences are truncated).

    Returns:
        Tensor of shape (n_sequences, max_len) with padded sequences.
    """
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)


def prepare_dataloader(
    panel: pd.DataFrame,
    max_len: int = 10,
    batch_size: int = 64,
) -> tuple[DataLoader, pd.DataFrame]:
    """Prepare DataLoader from panel data.

    Uses only the last period of each individual for cross-sectional estimation.

    Args:
        panel: Panel DataFrame with columns individual_id, treatment, outcome,
               career_sequence_history.
        max_len: Maximum sequence length for padding.
        batch_size: Batch size for the DataLoader.

    Returns:
        Tuple of (DataLoader, final cross-sectional DataFrame).
    """
    final = panel.groupby("individual_id").last().reset_index()

    sequences = pad_sequences(final["career_sequence_history"].tolist(), max_len)
    treatments = torch.tensor(final["treatment"].values, dtype=torch.long)
    outcomes = torch.tensor(final["outcome"].values, dtype=torch.float32)

    dataset = TensorDataset(sequences, treatments, outcomes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, final


def extract_embeddings(model, sequences: torch.Tensor) -> np.ndarray:
    """Extract embeddings from a trained model.

    Args:
        model: A trained embedding model with get_representation() method.
        sequences: Tensor of padded sequences.

    Returns:
        NumPy array of embeddings, shape (n_individuals, embedding_dim).
    """
    model.eval()
    with torch.no_grad():
        return model.get_representation(sequences).numpy()


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subheader(title: str):
    """Print a formatted sub-header."""
    print(f"\n  --- {title} ---")
