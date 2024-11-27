from dataclasses import dataclass
import torch
from torch import nn, Tensor

@dataclass
class Config():
    vocab_size:int=50304
    d_model:int=768
    n_heads:int=12
    n_layers:int=12
    d_ffn:int|None=None

class LayerState:
    # the recurrent neural network (RNN) state for a layer of RWKV5.2 
    def __init__(self, x, cfg:Config):
        B, T, C, H, K = x.size(0), x.size(1), cfg.d_model, cfg.n_heads, cfg.d_model // cfg.n_heads
        V = K
        # a (B,C) size tensor representing latest time mixer token embedding processed
        self.time_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)
        # an (B,H,K,V) size tensor representing a decaying token embedding memory for each head, where H=number_of_heads, K=key_dim_per_head, V=value_dim_per_head 
        self.kv_state = torch.zeros(B,H,K,V,dtype=torch.float32,device=x.device)
        # a (B,C) size tensor representing latest channel mixer token embedding processed
        self.channel_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)

# version 5.2 (RWKV version 5.1 has only per-head decays and u terms)
class RWKV(torch.nn.Module):
    def __init__(self, cfg:Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.ones(cfg.vocab_size, cfg.d_model) * 1e-4)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.layers = nn.ModuleList([Layer(cfg, layer_id) for layer_id in range(cfg.n_layers)])
        self.lm_head_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head_unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    # input tensor dimensions:
    #   x (B,T)
    def forward(self, x : Tensor, s : list[LayerState]|None = None):
        # calculate embeddings for each incoming token, then normalize them
        # see https://github.com/BlinkDL/SmallInitEmb for details on why we do this normalization
        # if you look at some literature on pre-genereated embeddings, you'll see that they are 
        #  often ideally considered to become unit length vectors around the hypersphere 
        #  so starting it as small noise while using a normalized version instantly causes this layout, 
        #  allowing them to initially move rapidly around the surface of that hypersphere, and later more slowly
        x = self.embed_norm(nn.functional.embedding(x, self.embed))

        s = s or [LayerState(x, self.cfg) for _ in range(self.cfg.n_layers)]

        # run each layer in succession, passing in the RNN state for that layer
        for layer_id, block in enumerate(self.layers):  # run each rwkv block
            x, s[layer_id] = block(x, s[layer_id])

        # normalize the output
        x = self.lm_head_norm(x)

        # unembed back to dictionary indices
        x = self.lm_head_unembed(x)

        return x, s

class Layer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.time_mixer = TimeMixer(cfg, layer_id)
        self.channel_mixer = ChannelMixer(cfg, layer_id)

    def forward(self, x : Tensor, s : LayerState):
        x, s.time_mixer_x_state, s.kv_state = self.time_mixer(x, s.time_mixer_x_state, s.kv_state)
        x, s.channel_mixer_x_state = self.channel_mixer(x, s.channel_mixer_x_state)
        return x, s

class TimeMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.cfg = cfg

        self.prenorm = nn.LayerNorm(cfg.d_model)

        self.tokenshift_receptance = nn.Parameter(torch.empty(cfg.d_model))
        self.receptance = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.tokenshift_key = nn.Parameter(torch.empty(cfg.d_model))
        self.key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.tokenshift_value = nn.Parameter(torch.empty(cfg.d_model))
        self.value = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.tokenshift_gate = nn.Parameter(torch.empty(cfg.d_model))
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # per-channel boost for current embedding
        self.bonus = nn.Parameter(torch.ones(cfg.n_heads, cfg.d_model//cfg.n_heads))

        # per-channel decay multipliers applied to kv_state at each timestep
        self.decay = nn.Parameter(torch.ones(cfg.n_heads, cfg.d_model//cfg.n_heads))
        
        self.group_norm = nn.GroupNorm(cfg.n_heads, cfg.d_model, eps=64e-5)

    @staticmethod
    def single_timestep(r, k, v, u, w, kv_state): 
        original_dtype = r.dtype

        B, H, K, V = kv_state.shape

        # transform inputs from BHK and put everything in float format for higher precision
        r = r.float().view(B, H, 1, K)
        w = w.float().view(B, H, K, 1)
        k = k.float().view(B, H, K, 1)
        v = v.float().view(B, H, 1, V)
        u = u.unsqueeze(-1).float() # 1HK1
        w = w.unsqueeze(-1).float() # 1HK1

        # start with the existing kv state
        y = kv_state        # BHKV
        # apply the u boost to the current k @ v and add it to that
        y = y + (k @ v) * u # BHKV * HK1 + BHKV = BHKV
        # apply receptance to that whole result
        out = r @ y         # BH1K @ BHKV = BH1V

        # finally, decay the kv state and add in the latest k @ v
        kv_state = kv_state * w         # BHKV
        kv_state = kv_state + (k @ v)   # BHKV * HK1 + BHKV = BHKV

        # remove an extra useless dimension from the output
        return out.squeeze(-2), kv_state # BHV, BHKV

    def forward(self, hidden_state_in : Tensor, x_state : Tensor, kv_state : Tensor): # x (B,T,C), x_state (B,C), kv_state (B,H,K,V)
        x = self.prenorm(hidden_state_in)
        x_state_out = x[:, -1]

        B, T, C, H, K = x.size(0), x.size(1), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads

        # we want the token embeddings shifted over by one towards the past
        # to get this, we take the last token embedding processed and append all but one of the current token embeddings to it
        # (the last token embedding processed is what's stored in the x_state)
        x_shifted_one_to_the_past = torch.cat((x_state.unsqueeze(-2), x[:,:-1]), dim=1)

        # token shift the incoming token embeddings for the receptance, key, value, and gate
        x_r = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_receptance)
        x_k = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_key)
        x_v = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_value)
        x_gate = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_gate)
                                           
        # the extra dimensions are being added here to enable matrix multiplications per timestep
        r = self.receptance(x_r).view(B,T,H,1,K) # BTH1K
        k = self.key(x_k).view(B,T,H,K,1) # BTHK1
        v = self.value(x_v).view(B,T,H,1,K) # BTH1K
        gate = self.gate(x_gate) # BTC

        # adding an extra dimension for convenience in multiplication later
        u = self.u.unsqueeze(-1) # HK1
        # this forces the decays to end up in the range 0...1 using a nicely differentiable function
        w = torch.exp(-torch.exp(self.w.float())).unsqueeze(-1) # HK1

        out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
        for t in range(T):
            out[:,t], kv_state = TimeMixer.single_timestep(r[:,t], k[:,t], v[:,t], u, w, kv_state)

        # apply group normalization to each head and recombine the heads
        out = self.group_norm(out.view(B*T, C)).view(B, T, C) # BTC

        # apply silu gate to the output
        out = out * nn.functional.silu(gate) # BTC

        # project the output
        out = self.Wproj_out(out) # BTC

        return hidden_state_in + out, x_state_out, kv_state

class ChannelMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.cfg = cfg
        self.prenorm = nn.LayerNorm(cfg.d_model)
        self.tokenshift_in = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.tokenshift_gate = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        d_ffn = cfg.d_ffn or int(cfg.d_model * 3.5)//32*32
        self.W_in = nn.Linear(cfg.d_model, d_ffn, bias=False)
        self.W_out = nn.Linear(d_ffn, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, hidden_state_in : Tensor, x_state : Tensor): # x (B,T,C), x_state (B,C)
        x = self.prenorm(hidden_state_in)
        x_state_out = x[:, -1]

        # token shift the incoming token embeddings for both the input projection and gate

        # token shift is like a a very efficient 1D convolution with kernel size 2, similar to undilated causal conv in WaveNet
        # this gives each head the ability to choose which parts of the time-series to pay attention to
        # it acts like a vertical forget gate between layers, choosing which parts of the recent past to accrue and which to ignore

        # we want the token embeddings shifted over by one towards the past
        # to get this, we take the last token embedding processed and append all but one of the current token embeddings to it
        # (the last token embedding processed is what's stored in the x_state)
        x_shifted_one_to_the_past = torch.cat((x_state.unsqueeze(-2), x[:,:-1]), dim=1)

        # token shift is just a learned linear interpolation between the current and previous token embeddings in the sequence
        # this is done by lerping between x and the shifted x we just calculated
        # note that it is a per-channel learned interpolation amount, not just a single value per head
        x_in = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_in)
        x_gate = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_gate)

        # project to 3x larger hidden dimension
        # this is 4x for vanilla transformers FFN, but it's typical to reduce it when adding new parameters 
        #  to allow comparison models with the same number of total parameters for a given d_model, n_layer
        # in rwkv5's case, if you drop it down to 3x you end up making up for the extra parameters in the gate
        # in rwkv6's case, if you drop it down to 3.5x you end up making up for the extra parameters in the gate
        hidden = self.W_in(x_in)

        # relu^2 activation function
        hidden = torch.square(torch.relu(hidden))

        # project back out to d_model
        out = self.W_out(hidden)

        # apply sigmoid gate
        gate = self.gate(x_gate)
        out = out * torch.sigmoid(gate)

        return hidden_state_in + out, x_state_out


if __name__ == "__main__":
    model = RWKV()
    model.forward(torch.ones(1,2,dtype=torch.long))
