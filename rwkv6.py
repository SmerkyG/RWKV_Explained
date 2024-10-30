import torch
from torch import nn, Tensor

# config, layer state and channel mixer are unchanged from RWKV5
from rwkv5_2 import Config, LayerState, ChannelMixer

# Only time mixer changes for RWKV6
class RWKV(torch.nn.Module):
    def __init__(self, cfg:Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.ones(cfg.n_embed, cfg.d_model) * 1e-4)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.layers = nn.ModuleList([Layer(cfg, layer_id) for layer_id in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.n_embed, bias=False)

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
        x = self.out_norm(x)

        # unembed back to dictionary indices
        x = self.unembed(x)

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
    
class LoRA_MLP(nn.Module):
    def __init__(self, dim:int, dim_hidden:int, init_value : Tensor|None = None):
        super().__init__()
        init_value = init_value if init_value is not None else torch.zeros(dim)
        self.base = nn.Parameter(init_value)
        self.W_a = nn.Linear(dim, dim_hidden, bias=False)
        self.W_b = nn.Linear(dim_hidden, dim, bias=False)

    def forward(self, x : Tensor): # x (B,T,C)
        # this is rwkv's version of low rank adaptation

        # the result has two components: a base value vector, and an offset
        # the offset is calculated by taking token shifted x and squeezing it through shrinking and expanding linear layers
        # using tanh as an activation in the middle of that sandwich
        # this offers greatly reduced cost in terms of both computation and parameters than a single dim->dim linear layer
        return self.base + self.W_b( nn.functional.tanh( self.W_a(x) ) )

# data-dependent linear interpolation
class DDLerp(nn.Module):
    def __init__(self, dim:int, dim_hidden:int):
        super().__init__()
        self.lora = LoRA_MLP(dim, dim_hidden)

    def forward(self, x_premixed: Tensor, x : Tensor, x_shifted_one_to_the_past : Tensor): # x (B,T,C)
        # a data-dependent linear interpolation between the current and previous token embeddings in the sequence
        # note that it is a per-channel interpolation amount, not just a single value per head

        # lora the interpolated value
        y = self.lora(x_premixed)

        # linearly interpolate again, this time based on the results of the lora
        y = torch.lerp(x, x_shifted_one_to_the_past, y)

        return y

class TimeMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.cfg = cfg

        self.time_mixer_prenorm = nn.LayerNorm(cfg.d_model)

        self.W_ddlerp_premix = nn.Parameter(torch.zeros(cfg.d_model))
        self.ddlerps = [DDLerp(cfg.d_model, 32) for _ in range(5)]
        self.decay_lora = LoRA_MLP(cfg.d_model, 64, torch.ones(cfg.d_model))
        self.W_proj_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_proj_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_proj_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_proj_g = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_proj_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # per-channel boost for current embedding
        self.u = nn.Parameter(torch.ones(cfg.n_heads, cfg.d_model//cfg.n_heads))

        self.group_norm = nn.GroupNorm(cfg.n_heads, cfg.d_model, eps=64e-5)

    @staticmethod
    def single_timestep(r, k, v, u, w, kv_state): 
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

    def forward(self, x : Tensor, x_state : Tensor, kv_state : Tensor): # x (B,T,C), x_state (B,C), kv_state (B,H,K,V)
        B, T, C, H, K = x.size(0), x.size(1), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads

        x = self.time_mixer_prenorm(x)

        # we want the token embeddings shifted over by one towards the past
        # to get this, we take the last token embedding processed and append all but one of the current token embeddings to it
        # (the last token embedding processed is what's stored in the x_state)
        x_shifted_one_to_the_past = torch.cat((x_state.unsqueeze(-2), x[:,:-1]), dim=1)

        # token shift the incoming token embeddings for the receptance, key, value, gate, and decay
        x_premixed = torch.lerp(x, x_shifted_one_to_the_past, self.W_ddlerp_premix)
        rx, kx, vx, gatex, wx = [ddlerp(x_premixed, x, x_shifted_one_to_the_past) for ddlerp in self.ddlerps]
                                           
        # project and separate out our vectors into attention heads
        # the extra dimensions are being added here to enable matrix multiplications per timestep
        r = self.W_proj_r(rx).view(B,T,H,1,K) # BTH1K
        k = self.W_proj_k(kx).view(B,T,H,K,1) # BTHK1
        v = self.W_proj_v(vx).view(B,T,H,1,K) # BTH1K
        gate = self.W_proj_g(gatex) # BTC

        # adding an extra dimension for convenience in multiplication later
        u = self.u.unsqueeze(-1) # HK1

        # per-channel data-dependent decays generated inexpensively via low rank adaptation
        w = self.decay_lora(wx)
        # separate out into attention heads
        w = w.view(T,H,K,1)
        # this forces the decays to end up in the range 0...1 using a nicely differentiable function
        w = torch.exp(-torch.exp(w)).to(u.dtype) # HK1

        out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
        for t in range(T):
            out[:,t], kv_state = TimeMixer.single_timestep(r[:,t], k[:,t], v[:,t], u, w[t], kv_state)

        # apply group normalization to each head and recombine the heads
        out = self.group_norm(out.view(B*T, C)).view(B, T, C) # BTC

        # apply silu gate to the output
        out = out * nn.functional.silu(gate) # BTC

        # project the output
        out = self.W_proj_out(out) # BTC

        return x + out, x[:, -1], kv_state

if __name__ == "__main__":
    model = RWKV()
    model.forward(torch.ones(1,2,dtype=torch.long))
