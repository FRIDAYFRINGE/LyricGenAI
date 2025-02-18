from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import tiktoken

app = Flask(__name__)
CORS(app)  

# ---------------------------
# Model Classes and Utilities
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads, in one go
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape for multi-head attention and transpose
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled dot-product attention with causal mask enabled
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024   # maximum context length
    vocab_size: int = 50257  # vocabulary size (for GPT-2)
    n_layer: int = 8         # number of transformer blocks
    n_head: int = 8          # number of attention heads
    n_embd: int = 256        # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: use the same weight matrix for input and output embeddings
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")
        # position ids
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        x = tok_emb + pos_emb
        # transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Collect parameters for weight decay versus no weight decay
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ---------------------------
# Model Initialization
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig()
model = GPT(config)
# Load model weights (update the path if necessary)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Initialize the tokenizer/encoder (using GPT-2's encoding)
enc = tiktoken.get_encoding("gpt2")

# ---------------------------
# Generation Function
# ---------------------------

def generate_text(prompt, max_length=100, num_return_sequences=1, top_k=50):
    """
    Generate text based on the prompt.
    :param prompt: Input text prompt.
    :param max_length: Maximum total length (in tokens) for the generated text.
    :param num_return_sequences: How many sequences to generate.
    :param top_k: Number of top tokens to sample from.
    :return: List of generated texts.
    """
    # Encode prompt into tokens
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    # Repeat for batch generation if needed
    tokens = tokens.repeat(num_return_sequences, 1).to(device)

    # Initialize a generator (seed can be adjusted or omitted for randomness)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(torch.seed())  # for reproducibility; remove for true randomness

    # Generation loop: keep appending tokens until max_length is reached
    with torch.no_grad():
        while tokens.size(1) < max_length:
            logits, _ = model(tokens)
            # Consider only the last token's logits for sampling
            logits = logits[:, -1, :]  # shape: (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # Top-k filtering: sample from the top k tokens
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
            next_token = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)

    # Decode each generated sequence (truncate to max_length tokens)
    results = []
    for i in range(num_return_sequences):
        token_list = tokens[i, :max_length].tolist()
        text = enc.decode(token_list)
        results.append(text)
    return results

# ---------------------------
# Flask API Endpoint
# ---------------------------

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    max_length = data.get("max_length", 100)
    num_return_sequences = data.get("num_return_sequences", 1)
    top_k = data.get("top_k", 50)
    
    generated_texts = generate_text(
        prompt, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences, 
        top_k=top_k
    )
    # If only one sequence is requested, return it as a string
    if num_return_sequences == 1:
        return jsonify({"generated": generated_texts[0]})
    else:
        return jsonify({"generated": generated_texts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)














































# @app.route('/generate', methods=['POST'])
# def generate_endpoint():
#     data = request.get_json(force=True)
#     prompt = data.get("prompt", "")
#     if not prompt:
#         return jsonify({"error": "No prompt provided"}), 400
#     max_length = data.get("max_length", 100)
#     num_return_sequences = data.get("num_return_sequences", 1)
#     top_k = data.get("top_k", 50)
    
#     generated_texts = generate_text(
#         prompt, 
#         max_length=max_length, 
#         num_return_sequences=num_return_sequences, 
#         top_k=top_k
#     )
#     # If only one sequence is requested, return it as a string
#     if num_return_sequences == 1:
#         return jsonify({"generated": generated_texts[0]})
#     else:
#         return jsonify({"generated": generated_texts})

# # ---------------------------
# # Main: Run the Flask App
# # ---------------------------

# if __name__ == '__main__':
#     # Run on all interfaces on port 5000 (adjust host/port as needed)
#     app.run(host='0.0.0.0', port=5000)
