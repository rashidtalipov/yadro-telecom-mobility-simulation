from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_cells: int
    numeric_dim: int
    target_mode: str = "flat"
    candidate_feature_dim: int = 5
    cell_embedding_dim: int = 16
    numeric_projection_dim: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MultitaskLstmPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_cell_index = config.num_cells
        self.cell_embedding = nn.Embedding(
            config.num_cells + 1,
            config.cell_embedding_dim,
            padding_idx=self.padding_cell_index,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(config.numeric_dim, config.numeric_projection_dim),
            nn.ReLU(),
            nn.LayerNorm(config.numeric_projection_dim),
        )
        self.backbone = nn.LSTM(
            input_size=config.cell_embedding_dim + config.numeric_projection_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.trigger_head = nn.Linear(config.hidden_size, 1)
        self.target_head = nn.Linear(config.hidden_size, config.num_cells)
        if config.target_mode == "candidate":
            self.candidate_feature_projection = nn.Sequential(
                nn.Linear(config.candidate_feature_dim, config.cell_embedding_dim),
                nn.ReLU(),
                nn.LayerNorm(config.cell_embedding_dim),
            )
            self.candidate_scorer = nn.Sequential(
                nn.Linear(
                    config.hidden_size + config.cell_embedding_dim + config.cell_embedding_dim,
                    config.hidden_size,
                ),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1),
            )

    def forward(
        self,
        numeric: torch.Tensor,
        serving_cell: torch.Tensor,
        candidate_cell: torch.Tensor | None = None,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cell_features = self.cell_embedding(serving_cell)
        numeric_features = self.numeric_projection(numeric)
        sequence_input = torch.cat([cell_features, numeric_features], dim=-1)
        lstm_output, _ = self.backbone(sequence_input)
        final_state = lstm_output[:, -1, :]
        shared = self.shared_head(final_state)
        outputs = {
            "trigger_logits": self.trigger_head(shared).squeeze(-1),
        }
        if self.config.target_mode == "flat":
            outputs["target_logits"] = self.target_head(shared)
            return outputs

        outputs["global_target_logits"] = self.target_head(shared)
        if candidate_cell is None or candidate_features is None:
            raise ValueError("candidate mode requires candidate_cell and candidate_features inputs")

        candidate_embedding = self.cell_embedding(candidate_cell)
        candidate_feature_state = self.candidate_feature_projection(candidate_features)
        repeated_shared = shared.unsqueeze(1).expand(-1, candidate_cell.shape[1], -1)
        scorer_input = torch.cat(
            [repeated_shared, candidate_embedding, candidate_feature_state],
            dim=-1,
        )
        candidate_logits = self.candidate_scorer(scorer_input).squeeze(-1)
        if candidate_mask is not None:
            candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e9)
        outputs["candidate_logits"] = candidate_logits
        return outputs
