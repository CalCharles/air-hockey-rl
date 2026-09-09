"""
ContextEncoder: maps a history of observations to a fixed-size context vector.

Input:  (batch, seq_len, obs_dim)   — last `context_len` raw observations
Output: (batch, context_dim)        — context vector appended to policy obs

Unlike the original Decision-Transformer style encoder, this module takes
only observations (no actions / returns / timesteps). It is co-trained with
the TD3 actor via the actor loss gradient.
"""


import transformers
import torch
import torch.nn as nn

from scripts.transformer.trajectory_model import TrajectoryModel
from transformers import GPT2Config, GPT2Model


class ContextEncoder(TrajectoryModel):
    """
    GPT-2 based context encoder.

    Embeds each observation in the history, adds learnable positional
    embeddings, runs them through a causal GPT-2 transformer, then
    mean-pools the output tokens to produce a single context vector.

    Parameters
    ----------
    obs_dim      : raw observation dimension (e.g. 30 for hist_len=2 policy)
    context_dim  : output context vector size (e.g. 8)
    context_len  : number of past observations to condition on (sequence length T)
    n_layer      : number of GPT-2 transformer layers
    n_head       : number of attention heads
    dropout      : dropout probability on embeddings + residuals
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int,
        context_len: int,
        n_layer: int = 3,   # was 2
        n_head: int = 1,    # was 2
        dropout: float = 0.1,
    ):
        # 32 is our hidden size
        hidden_size = 32 if 32 % n_head == 0 else n_head * ((context_dim + n_head - 1) // n_head)

        super().__init__(obs_dim=obs_dim, context_dim=context_dim, max_length=context_len)

        self.context_len = context_len
        self.hidden_size = hidden_size

        gpt_config = transformers.GPT2Config(
            vocab_size=1,           # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_ctx=context_len,
            n_positions=context_len,
            attn_pdrop=dropout,
            embd_pdrop=dropout,
            resid_pdrop=dropout,
        )

        self.transformer = GPT2Model(gpt_config)

        # --- Input projection ---
        self.embed_obs = nn.Linear(obs_dim, hidden_size)
        self.embed_pos = nn.Embedding(context_len, hidden_size)
        self.embed_ln  = nn.LayerNorm(hidden_size)

        # --- Output projection ---
        self.output_proj = nn.Linear(hidden_size, context_dim)


    def forward(self, obs_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        obs_sequence : (batch, seq_len, obs_dim)
            A batch of observation histories. seq_len must be <= context_len.
            Padding (zeros) at the start is fine — attention_mask handles it.

        Returns
        -------
        context_vector : (batch, context_dim)
        """
        batch_size, seq_len, _ = obs_sequence.shape
        device = obs_sequence.device

        # --- Embed observations ---
        obs_emb = self.embed_obs(obs_sequence)                    # (B, T, H)
        # TODO: need to check if this the shape here is correct

        # --- Add positional embeddings ---
        # positions are 0 … seq_len-1
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.embed_pos(positions)                       # (1, T, H)

        x = self.embed_ln(obs_emb + pos_emb)                     # (B, T, H)

        mask = (obs_sequence != 0).any(dim=-1).long()

        transformer_out = self.transformer(
            inputs_embeds=x,
            attention_mask=mask,
        )
        hidden = transformer_out["last_hidden_state"]             # (B, T, H)

        # Causal attention means the last token has already attended to
        # every earlier (unpadded) timestep, so it's a valid summary of
        # the whole history. Padding is left-aligned (at the start), so
        # position -1 is always the most recent real observation.
        last_hidden = hidden[:, -1, :] 
        
        context = self.output_proj(last_hidden)                        # (B, context_dim)
        return context
