"""Transformer-based fraud detection model."""

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds positional information to input sequences since Transformers
    don't have inherent notion of order.
    """

    def __init__(self, d_model: int, max_seq_length: int = 100, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerFraudDetector(nn.Module):
    """
    Transformer model for fraud detection.

    Architecture:
    - Linear projection to d_model dimensions
    - Positional encoding
    - Transformer encoder layers
    - Classification head
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 2,
        max_seq_length: int = 100,
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Number of input features
            d_model: Transformer dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            output_size: Number of output classes
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, output_size)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)

        # Take the last output (for sequence classification)
        last_out = x[:, -1, :]

        # Classification head
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerWithPooling(nn.Module):
    """
    Transformer with different pooling strategies for classification.

    Supports: mean pooling, max pooling, CLS token
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 2,
        max_seq_length: int = 100,
        pooling_strategy: str = "mean",  # Options: mean, max, cls
    ):
        """
        Initialize Transformer with pooling.

        Args:
            input_size: Number of input features
            d_model: Transformer dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            output_size: Number of output classes
            max_seq_length: Maximum sequence length
            pooling_strategy: Pooling strategy for classification
        """
        super().__init__()

        self.pooling_strategy = pooling_strategy
        self.d_model = d_model

        # CLS token if needed
        if pooling_strategy == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length + 1, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pooling.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Logits of shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)

        # Add CLS token if using CLS pooling
        if self.pooling_strategy == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Pool based on strategy
        if self.pooling_strategy == "cls":
            # Use CLS token
            pooled = x[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence length
            pooled = x.mean(dim=1)
        elif self.pooling_strategy == "max":
            # Max pooling over sequence length
            pooled, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Classification head
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(config: DictConfig) -> TransformerFraudDetector:
    """
    Create Transformer model from configuration.

    Args:
        config: Configuration object

    Returns:
        TransformerFraudDetector instance
    """
    model_config = config.model

    model = TransformerFraudDetector(
        input_size=model_config.input_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout,
        output_size=model_config.output_size,
        max_seq_length=model_config.get("max_seq_length", 100),
    )

    return model


def create_transformer_with_pooling(config: DictConfig, pooling: str = "mean") -> TransformerWithPooling:
    """
    Create Transformer with pooling from configuration.

    Args:
        config: Configuration object
        pooling: Pooling strategy

    Returns:
        TransformerWithPooling instance
    """
    model_config = config.model

    model = TransformerWithPooling(
        input_size=model_config.input_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout,
        output_size=model_config.output_size,
        max_seq_length=model_config.get("max_seq_length", 100),
        pooling_strategy=pooling,
    )

    return model
