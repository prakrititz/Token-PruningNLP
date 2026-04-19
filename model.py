"""
CopyEnhancedCodeT5: Pointer-Generator Network for Bugs2Fix Task
Integrates copy mechanism with CodeT5 base model
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load HF_TOKEN from .env file
def load_hf_token(env_file=".env"):
    """Load HF_TOKEN from .env file and set environment variable"""
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN"):
                    # Parse: HF_TOKEN = "token_value"
                    token_value = line.split("=")[1].strip().strip('"')
                    os.environ["HF_TOKEN"] = token_value
                    print(f"[+] Loaded HF_TOKEN from {env_file}")
                    return token_value
    return None


# Load token before importing/initializing models
load_hf_token()

# Disable HF hub conversion threads
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"


class CopyEnhancedCodeT5(nn.Module):
    """
    T5 model with copy mechanism (pointer-generator network).
    Loads Salesforce/codet5-base and adds pointer-generator network for improved compression.
    """
    
    def __init__(self, model_name="Salesforce/codet5-base"):
        super().__init__()
        
        os.environ["DISABLE_TELEMETRY"] = "1"
        
        print(f"Loading model from {model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_safetensors=False
        )
        
        # Model dimensions from model config
        self.hidden_dim = self.model.config.d_model
        self.vocab_size = self.model.config.vocab_size
        
        print(f"[+] Model loaded: hidden_dim={self.hidden_dim}, vocab_size={self.vocab_size}")
        
        # Linear layer for generation probability gate (copy mechanism)
        self.W_gen = nn.Linear(self.hidden_dim * 2, 1)
        
    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None):
        """
        Forward pass with copy mechanism.
        
        Args:
            input_ids: [batch, src_len] - encoder input token IDs
            decoder_input_ids: [batch, tgt_len] - decoder input token IDs
            attention_mask: [batch, src_len] - encoder attention mask
            decoder_attention_mask: [batch, tgt_len] - decoder attention mask
            
        Returns:
            P_final: [batch, tgt_len, vocab_size] - final probability distribution
        """
        
        batch_size = input_ids.shape[0]
        tgt_len = decoder_input_ids.shape[1]
        device = input_ids.device
        
        # ===== STEP 1: Get CodeT5 outputs with attention and hidden states =====
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract logits from model
        # logits: [batch, tgt_len, vocab_size]
        logits = outputs.logits
        
        # Extract encoder hidden states from final layer
        # encoder_hidden_states: [batch, src_len, hidden_dim]
        encoder_hidden_states = outputs.encoder_hidden_states[-1]
        
        # Extract decoder hidden states from final layer
        # decoder_hidden_states: [batch, tgt_len, hidden_dim]
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        s_t = decoder_hidden_states
        
        # Extract cross-attention from final decoder layer
        # cross_attentions shape: [batch, num_heads, tgt_len, src_len]
        cross_attentions = outputs.cross_attentions[-1]
        # Average over attention heads: [batch, tgt_len, src_len]
        A = cross_attentions.mean(dim=1)
        
        # ===== STEP 2: Compute Context Vector h_t* =====
        # Context vector = attention-weighted sum of encoder hidden states
        # A: [batch, tgt_len, src_len]
        # encoder_hidden_states: [batch, src_len, hidden_dim]
        # h_t_star: [batch, tgt_len, hidden_dim]
        h_t_star = torch.bmm(A, encoder_hidden_states)
        
        # ===== STEP 3: Compute Generation Probability p_gen =====
        # Concatenate context vector and decoder hidden state
        # concat: [batch, tgt_len, hidden_dim * 2]
        concat = torch.cat([h_t_star, s_t], dim=-1)
        
        # Pass through W_gen and apply sigmoid
        # p_gen: [batch, tgt_len, 1]
        p_gen = torch.sigmoid(self.W_gen(concat))
        
        # ===== STEP 4: Compute Vocabulary Distribution P_vocab =====
        # Apply softmax to logits
        # P_vocab: [batch, tgt_len, vocab_size]
        P_vocab = torch.softmax(logits, dim=-1)
        
        # ===== STEP 5: Compute Copy Distribution P_copy =====
        # Initialize zero tensor for copy distribution (match A's dtype for AMP compatibility)
        # P_copy: [batch, tgt_len, vocab_size]
        P_copy = torch.zeros(batch_size, tgt_len, self.vocab_size, device=device, dtype=A.dtype)
        
        # Scatter attention weights into vocabulary dimension — VECTORIZED
        # Old code used a double for-loop (B × T python calls) which was ~500x slower.
        # Instead, expand input_ids to [B, T, S] so scatter_add_ runs in one fused op.
        # input_ids: [B, S] → [B, 1, S] → [B, T, S]
        idx = input_ids.unsqueeze(1).expand(-1, tgt_len, -1)   # [B, T, S]
        # A: [B, T, S]  — attention weights to scatter into vocab positions
        P_copy.scatter_add_(dim=2, index=idx, src=A)
        
        # ===== STEP 6: Combine using Generation Gate =====
        # Final distribution: P(y) = p_gen * P_vocab + (1 - p_gen) * P_copy
        # p_gen: [batch, tgt_len, 1]
        # P_vocab: [batch, tgt_len, vocab_size]
        # P_copy: [batch, tgt_len, vocab_size]
        # P_final: [batch, tgt_len, vocab_size]
        P_final = p_gen * P_vocab + (1 - p_gen) * P_copy
        
        return P_final


if __name__ == "__main__":
    print("Loading CopyEnhancedCodeT5 (Salesforce/codet5-large)...")
    model = CopyEnhancedCodeT5()
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    input_ids = torch.randint(0, model.vocab_size, (batch_size, src_len))
    decoder_input_ids = torch.randint(0, model.vocab_size, (batch_size, tgt_len))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    
    print("\n[+] Running forward pass...")
    with torch.no_grad():
        P_final = model(input_ids, decoder_input_ids)
    
    print(f"\n[+] Output P(y) shape: {P_final.shape}")
    print(f"    Expected: [{batch_size}, {tgt_len}, {model.vocab_size}]")
    print(f"    Match: {P_final.shape == (batch_size, tgt_len, model.vocab_size)}")
    
    # Verify probabilities sum to 1
    prob_sum = P_final.sum(dim=-1)
    print(f"\n[+] Probability sum (should be ~1.0):")
    print(f"    Min: {prob_sum.min().item():.6f}")
    print(f"    Max: {prob_sum.max().item():.6f}")
    print(f"    Mean: {prob_sum.mean().item():.6f}")
    
    print("\n[✓] CopyEnhancedCodeT5 successfully initialized and forward pass completed!")
