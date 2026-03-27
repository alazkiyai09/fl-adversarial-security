"""LSTM-based fraud detection model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class LSTMFraudDetector(nn.Module):
    """
    LSTM model for fraud detection on transaction sequences.

    Architecture:
    - LSTM layer(s) for sequence modeling
    - Dropout for regularization
    - Fully connected layers for classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output classes (2 for binary)
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_output_size // 2, output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Optional hidden state from previous forward pass

        Returns:
            Logits of shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last output (for sequence classification)
        last_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0) hidden states
        """
        num_directions = 2 if self.bidirectional else 1

        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

        return h_0, c_0


class AttentionLSTM(LSTMFraudDetector):
    """
    LSTM with attention mechanism for fraud detection.

    Attention helps the model focus on important transactions in the sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
        bidirectional: bool = False,
        attention_heads: int = 4,
    ):
        """
        Initialize Attention LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output classes
            bidirectional: Use bidirectional LSTM
            attention_heads: Number of attention heads
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # Update fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Optional hidden state

        Returns:
            Logits of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Take the last output after attention
        last_out = attn_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def create_lstm_model(config: DictConfig) -> LSTMFraudDetector:
    """
    Create LSTM model from configuration.

    Args:
        config: Configuration object

    Returns:
        LSTMFraudDetector instance
    """
    model_config = config.model

    model = LSTMFraudDetector(
        input_size=model_config.input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        output_size=model_config.output_size,
        bidirectional=model_config.get("bidirectional", False),
    )

    return model


def create_attention_lstm_model(config: DictConfig) -> AttentionLSTM:
    """
    Create Attention LSTM model from configuration.

    Args:
        config: Configuration object

    Returns:
        AttentionLSTM instance
    """
    model_config = config.model

    model = AttentionLSTM(
        input_size=model_config.input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        output_size=model_config.output_size,
        bidirectional=model_config.get("bidirectional", False),
        attention_heads=model_config.get("attention_heads", 4),
    )

    return model
