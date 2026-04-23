"""
copy_codet5.py - CodeT5 with Copy Mechanism

Wraps the standard T5ForConditionalGeneration model with a copy mechanism
(pointer-generator network) as described in Section 4.2 of the paper.

Architecture modifications:
1. Extended vocabulary with special tokens (<ASSERTION>, <Ratio>, etc.)
2. Copy mechanism that blends vocabulary and copy distributions
3. Uses the last decoder cross-attention layer for copy distribution

References:
    - See et al., 2017 (Pointer-Generator Networks)
    - Zhang et al., 2021 (Point, Disambiguate and Copy)
    - Wang et al., 2021 (CodeT5)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.model.copy_module import CopyModule


class CopyCodeT5(nn.Module):
    """
    CodeT5 augmented with a copy mechanism.

    During decoding, at each step the model:
    1. Runs the standard T5 decoder to get hidden states and cross-attention
    2. Computes generation probability p_gen via the copy module
    3. Blends the vocabulary distribution with the copy distribution
    4. Outputs the final token probabilities

    The model can be used with or without the copy mechanism (controlled by
    `use_copy` flag) to enable ablation studies.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/codet5-base",
        tokenizer=None,
        use_copy: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for CodeT5.
            tokenizer: Extended tokenizer (with special tokens already added).
            use_copy: Whether to use the copy mechanism.
        """
        super().__init__()

        self.use_copy = use_copy
        self.model_name = model_name

        # Load base T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        # Resize token embeddings if tokenizer has additional tokens
        if tokenizer is not None:
            self.t5.resize_token_embeddings(len(tokenizer))

        # Get hidden size from config
        self.hidden_size = self.t5.config.d_model  # 1024 for codet5-large

        # Initialize copy module
        if self.use_copy:
            self.copy_module = CopyModule(hidden_size=self.hidden_size)

        # Store vocab size
        self.vocab_size = self.t5.config.vocab_size
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with copy mechanism.

        Args:
            input_ids: Source token IDs. Shape: (B, T_src)
            attention_mask: Source attention mask. Shape: (B, T_src)
            decoder_input_ids: Decoder input IDs. Shape: (B, T_tgt)
            labels: Target token IDs for loss computation. Shape: (B, T_tgt)

        Returns:
            Dictionary with 'loss', 'logits', and optionally 'p_gen'.
        """
        # Shift labels to create decoder_input_ids if not provided
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.t5._shift_right(labels)

        # Run the full T5 model with cross-attention output enabled
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=None,  # We compute loss ourselves when using copy
            output_attentions=True,  # Need cross-attention weights
            output_hidden_states=True,
            use_cache=False, # Required for gradient checkpointing
            return_dict=True,
        )

        if not self.use_copy:
            # Standard T5 behavior - just compute loss normally
            if labels is not None:
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                return {"loss": loss, "logits": logits}
            return {"loss": None, "logits": outputs.logits}

        # === Copy mechanism path ===

        # Get decoder hidden states (last layer)
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        # decoder_hidden_states: (B, T_tgt, H)

        # Get cross-attention from the last decoder layer
        # cross_attentions is a tuple of (num_layers,) each (B, num_heads, T_tgt, T_src)
        last_cross_attention = outputs.cross_attentions[-1]
        # Average over attention heads
        cross_attention_weights = last_cross_attention.mean(dim=1)
        # cross_attention_weights: (B, T_tgt, T_src)

        # Normalize attention weights (ensure they sum to 1 over source)
        cross_attention_weights = cross_attention_weights / (
            cross_attention_weights.sum(dim=-1, keepdim=True) + 1e-12
        )

        # Get encoder hidden states
        encoder_hidden_states = outputs.encoder_last_hidden_state
        # encoder_hidden_states: (B, T_src, H)

        # Get vocab logits from the LM head
        vocab_logits = outputs.logits
        # vocab_logits: (B, T_tgt, V)

        # Apply copy module to get final log-probabilities
        log_probs = self.copy_module(
            decoder_hidden_states=decoder_hidden_states,
            cross_attention_weights=cross_attention_weights,
            encoder_hidden_states=encoder_hidden_states,
            vocab_logits=vocab_logits,
            source_ids=input_ids,
            vocab_size=self.vocab_size,
        )
        # log_probs: (B, T_tgt, V)

        # Compute loss if labels provided (Eq. 10: Cross-Entropy Loss)
        loss = None
        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=-100)
            loss = loss_fct(
                log_probs.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": log_probs,  # These are log-probs, not raw logits
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_beams: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate compressed code using greedy/beam search with copy mechanism.

        For simplicity, we use greedy decoding when copy mechanism is enabled.
        Without copy, we delegate to T5's built-in generate.

        Args:
            input_ids: Source token IDs. Shape: (B, T_src)
            attention_mask: Source attention mask. Shape: (B, T_src)
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.

        Returns:
            Generated token IDs. Shape: (B, T_gen)
        """
        if not self.use_copy:
            return self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs,
            )

        # Greedy decoding with copy mechanism
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get encoder outputs (compute once)
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Start with decoder start token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track generated sequences
        generated = decoder_input_ids.clone()
        eos_token_id = self.t5.config.eos_token_id

        for step in range(max_length - 1):
            # Run decoder
            decoder_outputs = self.t5.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            # Get last token's hidden state and attention
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
            # last_hidden: (B, 1, H)

            # Get cross-attention for last position from last layer
            last_cross_attn = decoder_outputs.cross_attentions[-1][:, :, -1:, :]
            # last_cross_attn: (B, num_heads, 1, T_src)
            cross_attn_avg = last_cross_attn.mean(dim=1)
            # cross_attn_avg: (B, 1, T_src)

            # Normalize
            cross_attn_avg = cross_attn_avg / (
                cross_attn_avg.sum(dim=-1, keepdim=True) + 1e-12
            )

            # Get vocab logits for last position
            lm_logits = self.t5.lm_head(last_hidden)
            # lm_logits: (B, 1, V_original)

            # Pad logits if vocab was extended
            if lm_logits.shape[-1] < self.vocab_size:
                padding = torch.zeros(
                    batch_size, 1, self.vocab_size - lm_logits.shape[-1],
                    dtype=lm_logits.dtype, device=device
                )
                lm_logits = torch.cat([lm_logits, padding], dim=-1)

            # Apply copy mechanism
            log_probs = self.copy_module(
                decoder_hidden_states=last_hidden,
                cross_attention_weights=cross_attn_avg,
                encoder_hidden_states=encoder_hidden_states,
                vocab_logits=lm_logits,
                source_ids=input_ids,
                vocab_size=self.vocab_size,
            )
            # log_probs: (B, 1, V)

            # Greedy: take argmax
            next_token = log_probs[:, -1, :].argmax(dim=-1, keepdim=True)
            # next_token: (B, 1)

            generated = torch.cat([generated, next_token], dim=1)
            decoder_input_ids = generated

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def save_pretrained(self, save_dir: str):
        """Save the model (T5 + copy module)."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save T5 base
        self.t5.save_pretrained(save_dir)

        # Save copy module separately
        if self.use_copy:
            copy_path = os.path.join(save_dir, "copy_module.pt")
            torch.save(self.copy_module.state_dict(), copy_path)

        # Save config
        config = {
            "model_name": self.model_name,
            "use_copy": self.use_copy,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }
        import json
        config_path = os.path.join(save_dir, "copy_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str, tokenizer=None):
        """Load a saved model."""
        import json

        config_path = os.path.join(load_dir, "copy_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model (this loads the T5 weights from load_dir)
        model = cls(
            model_name=load_dir,  # Load T5 from the saved directory
            tokenizer=tokenizer,
            use_copy=config["use_copy"],
        )

        # Load copy module weights
        if config["use_copy"]:
            copy_path = os.path.join(load_dir, "copy_module.pt")
            if os.path.exists(copy_path):
                model.copy_module.load_state_dict(
                    torch.load(copy_path, map_location="cpu")
                )

        return model


def create_model(
    model_name: str = "Salesforce/codet5-base",
    use_copy: bool = True,
    tokenizer=None,
    gradient_checkpointing: bool = True,
) -> CopyCodeT5:
    """
    Factory function to create the CopyCodeT5 model.

    Args:
        model_name: HuggingFace model identifier.
        use_copy: Whether to include the copy mechanism.
        tokenizer: Extended tokenizer.
        gradient_checkpointing: Enable gradient checkpointing for memory savings.

    Returns:
        Initialized CopyCodeT5 model.
    """
    model = CopyCodeT5(
        model_name=model_name,
        tokenizer=tokenizer,
        use_copy=use_copy,
    )

    if gradient_checkpointing:
        model.t5.gradient_checkpointing_enable()
        print("[Model] Gradient checkpointing enabled")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    copy_params = sum(p.numel() for p in model.copy_module.parameters()) if use_copy else 0

    print(f"[Model] Created CopyCodeT5 ({'with' if use_copy else 'without'} copy)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Copy module parameters: {copy_params:,}")
    print(f"  Vocabulary size: {model.vocab_size}")

    return model


if __name__ == "__main__":
    import os
    from src.tokenizer_utils import get_extended_tokenizer

    print("=== Testing CopyCodeT5 ===\n")

    # Use base for quick testing
    test_model_name = "Salesforce/codet5-base"

    # Get tokenizer
    tokenizer = get_extended_tokenizer(test_model_name)

    # Create model
    model = create_model(
        model_name=test_model_name,
        use_copy=True,
        tokenizer=tokenizer,
        gradient_checkpointing=False,
    )

    # Test forward pass
    sample_input = "<BUGS2FIX> <Ratio> 0.3 </Ratio> <Compress> public static void main ( ) { } </Compress>"
    sample_target = "<Compress> public static void main ( ) </Compress>"

    inputs = tokenizer(sample_input, return_tensors="pt", max_length=128, truncation=True)
    targets = tokenizer(sample_target, return_tensors="pt", max_length=128, truncation=True)

    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Target shape: {targets['input_ids'].shape}")

    # Forward pass
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=targets["input_ids"],
    )
    print(f"\nLoss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")

    # Test generation
    gen_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
    )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
    print(f"\nGenerated: {gen_text}")
    print("\n=== Test Complete ===")
