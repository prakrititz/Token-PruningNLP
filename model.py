"""
CopyEnhancedCodeT5: Pointer-Generator Network for Bugs2Fix Task
Integrates copy mechanism with CodeT5 base model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AddedToken


class CopyEnhancedCodeT5(nn.Module):
    """
    T5 model with copy mechanism (pointer-generator network).
    Loads Salesforce/codet5-large and adds pointer-generator network for improved compression.
    """
    
    def __init__(self, model_name="Salesforce/codet5-large"):
        super().__init__()
        
        # Load base CodeT5 model
        print(f"Loading model from {model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add special tokens for compression task
        try:
            special_tokens_dict = {
                "additional_special_tokens": ["<BUGS2FIX>", "<Ratio>", "</Ratio>", "<Compress>", "</Compress>"]
            }
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_toks} special tokens")
            
            # Resize model embeddings to account for new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            print(f"Warning: Could not add special tokens: {e}")
        
        # Model dimensions
        self.hidden_dim = self.model.config.d_model  # 1024 for codet5-large
        self.vocab_size = len(self.tokenizer)
        
        print(f"Model hidden_dim: {self.hidden_dim}, vocab_size: {self.vocab_size}")
        
        # Linear layer for generation probability gate (copy mechanism)
        # W_gen maps [context_vector + decoder_hidden] to probability
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
        # Initialize zero tensor for copy distribution
        # P_copy: [batch, tgt_len, vocab_size]
        P_copy = torch.zeros(batch_size, tgt_len, self.vocab_size, device=device)
        
        # Scatter attention weights into vocabulary dimension
        for b in range(batch_size):
            for t in range(tgt_len):
                # A[b, t]: [src_len] - attention weights over source tokens
                # input_ids[b]: [src_len] - source token indices
                P_copy[b, t].scatter_add_(0, input_ids[b], A[b, t])
        
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
    
    # Sample code snippets for compression
    buggy_code = """
    public void processData(List<String> items) {
        for (String item : items) {
            System.out.println(item);
            int length = item.length();
        }
    }
    """
    
    fixed_code = """
    public void processData(List<String> items) {
        for (String item : items) {
            if (item != null) {
                System.out.println(item);
            }
        }
    }
    """
    
    # Tokenize
    print("\nTokenizing input...")
    inputs = model.tokenizer(
        buggy_code,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    decoder_inputs = model.tokenizer(
        fixed_code,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    print(f"Input shape: {inputs.input_ids.shape}")
    print(f"Decoder input shape: {decoder_inputs.input_ids.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    P_final = model(
        input_ids=inputs.input_ids,
        decoder_input_ids=decoder_inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_attention_mask=decoder_inputs.attention_mask
    )
    
    # Verify output
    print(f"\nOutput P(y) shape: {P_final.shape}")
    batch_size, tgt_len, vocab_size = P_final.shape
    print(f"Expected: [batch_size={batch_size}, tgt_len={tgt_len}, vocab_size={vocab_size}]")
    
    # Check probabilities sum to 1
    prob_sum = P_final.sum(dim=-1)
    print(f"\nProbability sum (should be ~1.0):")
    print(f"  Min: {prob_sum.min().item():.4f}")
    print(f"  Max: {prob_sum.max().item():.4f}")
    print(f"  Mean: {prob_sum.mean().item():.4f}")
    
    print("\n✓ CopyEnhancedCodeT5 successfully initialized and forward pass completed!")
