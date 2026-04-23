"""
copy_module.py - Copy Mechanism for CodeT5

Implements the Pointer-Generator / Copy mechanism from Section 4.2 of the paper.
This module computes the probability of copying tokens directly from the input
vs generating from the vocabulary.

Key equations from the paper:
    Eq. 5: h*_t = Σ_i a^t_i · h_i                    (context vector)
    Eq. 6: p_gen = σ(W_gen · [h*_t, s_t] + b_gen)     (generation probability)
    Eq. 7: P_copy(y) = Σ_{i: x_i = y} a^t_i           (copy distribution)
    Eq. 8: P_vocab(y) = Softmax(W_head · s_t + b_head) (vocab distribution)
    Eq. 9: P(y) = p_gen · P_vocab(y) + (1-p_gen) · P_copy(y) (final distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CopyModule(nn.Module):
    """
    Copy mechanism that computes the probability of generating from vocabulary
    vs copying from the source input.

    The module takes:
    - Cross-attention weights (from the last decoder layer)
    - Context vector (weighted sum of encoder hidden states)
    - Decoder hidden state

    And produces:
    - p_gen: probability of generating from vocabulary
    - copy_distribution: probability distribution over source tokens for copying
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Hidden dimension of the model (1024 for codet5-large).
        """
        super().__init__()

        # Linear layer to compute p_gen from [context_vector, decoder_state]
        # Input: concatenation of context vector h*_t and decoder state s_t
        # Output: scalar (before sigmoid)
        self.gen_linear = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,    # s_t: (batch, tgt_len, hidden)
        cross_attention_weights: torch.Tensor,  # A: (batch, tgt_len, src_len)
        encoder_hidden_states: torch.Tensor,    # h: (batch, src_len, hidden)
        vocab_logits: torch.Tensor,             # (batch, tgt_len, vocab_size)
        source_ids: torch.Tensor,               # (batch, src_len) - input token IDs
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Compute the final output distribution combining vocab and copy distributions.

        Args:
            decoder_hidden_states: Decoder outputs s_t. Shape: (B, T_tgt, H)
            cross_attention_weights: Cross-attention weights A. Shape: (B, T_tgt, T_src)
            encoder_hidden_states: Encoder outputs h. Shape: (B, T_src, H)
            vocab_logits: Raw logits from the LM head. Shape: (B, T_tgt, V)
            source_ids: Input token IDs. Shape: (B, T_src)
            vocab_size: Size of the extended vocabulary.

        Returns:
            Final log-probability distribution. Shape: (B, T_tgt, V)
        """
        batch_size, tgt_len, hidden_size = decoder_hidden_states.shape
        src_len = encoder_hidden_states.shape[1]

        # Step 1: Compute context vector h*_t (Eq. 5)
        # h*_t = Σ_i a^t_i · h_i
        # cross_attention_weights: (B, T_tgt, T_src)
        # encoder_hidden_states: (B, T_src, H)
        context_vector = torch.bmm(cross_attention_weights, encoder_hidden_states)
        # context_vector: (B, T_tgt, H)

        # Step 2: Compute generation probability p_gen (Eq. 6)
        # p_gen = σ(W_gen · [h*_t, s_t] + b_gen)
        gen_input = torch.cat([context_vector, decoder_hidden_states], dim=-1)
        # gen_input: (B, T_tgt, 2*H)
        p_gen = torch.sigmoid(self.gen_linear(gen_input))
        # p_gen: (B, T_tgt, 1)

        # Step 3: Compute vocab distribution P_vocab (Eq. 8)
        # P_vocab(y) = Softmax(logits)
        vocab_dist = F.softmax(vocab_logits, dim=-1)
        # vocab_dist: (B, T_tgt, V)

        # Step 4: Compute copy distribution P_copy (Eq. 7)
        # P_copy(y) = Σ_{i: x_i = y} a^t_i
        # We scatter-add attention weights into a vocab-sized tensor
        copy_dist = torch.zeros(
            batch_size, tgt_len, vocab_size,
            dtype=vocab_dist.dtype, device=vocab_dist.device
        )
        # Expand source_ids for scatter: (B, T_src) -> (B, T_tgt, T_src)
        source_ids_expanded = source_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        # Scatter-add attention weights to corresponding source token positions
        copy_dist.scatter_add_(2, source_ids_expanded, cross_attention_weights)
        # copy_dist: (B, T_tgt, V)

        # Step 5: Final distribution (Eq. 9)
        # P(y) = p_gen · P_vocab(y) + (1 - p_gen) · P_copy(y)
        final_dist = p_gen * vocab_dist + (1.0 - p_gen) * copy_dist
        # final_dist: (B, T_tgt, V)

        # Add small epsilon to avoid log(0)
        final_dist = final_dist + 1e-12

        # Return log probabilities for cross-entropy loss
        return torch.log(final_dist)


class CopyModuleWithGating(CopyModule):
    """
    Extended copy module with separate gating for context vector
    (more flexible than the basic version).
    """

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)

        # Additional projection for the context vector
        self.context_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        cross_attention_weights: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vocab_logits: torch.Tensor,
        source_ids: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Same interface as parent, with additional context projection."""
        batch_size, tgt_len, hidden_size = decoder_hidden_states.shape

        # Context vector with projection
        context_vector = torch.bmm(cross_attention_weights, encoder_hidden_states)
        context_vector = self.context_proj(context_vector)

        # Generation probability
        gen_input = torch.cat([context_vector, decoder_hidden_states], dim=-1)
        p_gen = torch.sigmoid(self.gen_linear(gen_input))

        # Vocab distribution
        vocab_dist = F.softmax(vocab_logits, dim=-1)

        # Copy distribution
        copy_dist = torch.zeros(
            batch_size, tgt_len, vocab_size,
            dtype=vocab_dist.dtype, device=vocab_dist.device
        )
        source_ids_expanded = source_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        copy_dist.scatter_add_(2, source_ids_expanded, cross_attention_weights)

        # Final distribution
        final_dist = p_gen * vocab_dist + (1.0 - p_gen) * copy_dist
        final_dist = final_dist + 1e-12

        return torch.log(final_dist)
