import torch
from torch import nn, Tensor

class RWKV5_Layer_State:
    # the recurrent neural network (RNN) state for a layer of RWKV5.2 
    def __init__(self, x):
        # a (B,C) size tensor representing latest time mixer token embedding processed
        self.time_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)
        # an (B,H,K,V) size tensor representing a decaying token embedding memory for each head, where H=number_of_heads, K=key_dim_per_head, V=value_dim_per_head 
        self.kv_state = torch.zeros(B,H,K,V,dtype=x.dtype,device=x.device)
        # a (B,C) size tensor representing latest channel mixer token embedding processed
        self.channel_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)

# version 5.2 (RWKV version 5.1 has only per-head decays and u terms)
class RWKV5(torch.nn.Module):
    def __init__(self, n_embed:int=50304, d_model:int=768, n_heads:int=12, n_layers:int=12):
        super().__init__()
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.embed = nn.Parameter(torch.ones(n_embed, d_model) * 1e-4)
        self.embed_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([RWKV5_Layer(self, layer_id) for layer_id in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    # input tensor dimensions:
    #   x (B,T)
    def forward(self, x : Tensor, s : list[RWKV5_Layer_State]|None = None):
        s = s or [RWKV5_Layer_State(x) for _ in range(self.n_layers)]

        # calculate embeddings for each incoming token, then normalize them
        # see https://github.com/BlinkDL/SmallInitEmb for details on why we do this normalization
        # if you look at some literature on pre-genereated embeddings, you'll see that they are 
        #  often ideally considered to become unit length vectors around the hypersphere 
        #  so starting it as small noise while using a normalized version instantly causes this layout, 
        #  allowing them to initially move rapidly around the surface of that hypersphere, and later more slowly
        x = self.embed_norm(nn.functional.embedding(x, self.embed))

        # run each layer in succession, passing in the RNN state for that layer
        for layer_id, block in enumerate(self.layers):  # run each rwkv block
            x, s[layer_id] = block(x, s[layer_id])

        # normalize the output
        x = self.out_norm(x)

        # unembed back to dictionary indices via weight tying of the embedding weights
        # this is just the opposite transformation as the original embedding calculation
        x = nn.functional.linear(x, self.embed)

        return x, s

class RWKV5_Layer(nn.Module):
    def __init__(self, rwkv:RWKV5, layer_id:int):
        super().__init__()
        self.time_mixer = RWKV5_TimeMixer(rwkv, layer_id)
        self.channel_mixer = RWKV5_ChannelMixer(rwkv, layer_id)

    def forward(self, x : Tensor, s : RWKV5_Layer_State):
        x, s.time_mixer_x_state, s.kv_state = self.time_mixer(x, s.time_mixer_x_state, s.kv_state)
        x, s.channel_mixer_x_state = self.channel_mixer(x, s.channel_mixer_x_state)
        return x, s

def token_shift(x, x_state, *Wlerps):
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
    return [torch.lerp(x, x_shifted_one_to_the_past, Wlerp(x)) for Wlerp in Wlerps]

class RWKV5_TimeMixer(nn.Module):
    def __init__(self, rwkv:RWKV5, layer_id:int):
        super().__init__()
        self.rwkv = rwkv

        self.time_mixer_prenorm = nn.LayerNorm(rwkv.d_model)

        self.Wlerp_r = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wproj_r = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wlerp_k = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wproj_k = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wlerp_v = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wproj_v = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wlerp_g = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wproj_g = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wproj_out = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)

        # per-channel boost for current embedding
        self.u = nn.Parameter(torch.ones(rwkv.n_heads, rwkv.d_model//rwkv.n_heads))

        # per-channel decay multipliers applied to kv_state at each timestep
        self.w = nn.Parameter(torch.ones(rwkv.n_heads, rwkv.d_model//rwkv.n_heads))
        
        self.group_norm = nn.GroupNorm(rwkv.n_heads, rwkv.d_model, eps=64e-5)

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
        B, T, C, H, K = x.size(0), x.size(1), self.rwkv.d_model, self.rwkv.n_heads, self.rwkv.d_model // self.rwkv.n_heads

        x = self.time_mixer_prenorm(x)

        # token shift the incoming token embeddings for the receptance, key, value, and gate
        rx, kx, vx, gatex = token_shift(x, x_state, self.Wlerp_r, self.Wlerp_k, self.Wlerp_v, self.Wlerp_g)
                                           
        # the extra dimensions are being added here to enable matrix multiplications per timestep
        r = self.Wproj_r(rx).view(B,T,H,1,K) # BTH1K
        k = self.Wproj_k(kx).view(B,T,H,K,1) # BTHK1
        v = self.Wproj_v(vx).view(B,T,H,1,K) # BTH1K
        gate = self.Wproj_g(gatex) # BTC

        # adding an extra dimension for convenience in multiplication later
        u = self.u.unsqueeze(-1) # HK1
        # this forces the decays to end up in the range 0...1 using a nicely differentiable function
        w = torch.exp(-torch.exp(self.w)).unsqueeze(-1).to(u.dtype) # HK1

        out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
        for t in range(T):
            out[:,t], s = RWKV5_TimeMixer.single_timestep(r[:,t], k[:,t], v[:,t], u, w, kv_state)

        # apply group normalization to each head
        out = self.group_norm(out.view(B*T, C)).view(B, T, C) # BTC

        # apply silu gate to the output
        out = out * nn.functional.silu(gate) # BTC

        # project the output
        out = self.Wproj_out(out) # BTC

        return x + out, x[:, -1], kv_state

class RWKV5_ChannelMixer(nn.Module):
    def __init__(self, rwkv:RWKV5, layer_id:int):
        super().__init__()
        self.rwkv = rwkv
        self.prenorm = nn.LayerNorm(rwkv.d_model)
        self.Win = nn.Linear(rwkv.d_model, rwkv.d_model * 3, bias=False)
        self.Wout = nn.Linear(rwkv.d_model * 3, rwkv.d_model, bias=False)
        self.Wlerp_in = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wlerp_g = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)
        self.Wgate = nn.Linear(rwkv.d_model, rwkv.d_model, bias=False)

    def forward(self, x : Tensor, x_state : Tensor): # x (B,T,C), x_state (B,C)
        x = self.prenorm(x)

        # token shift the incoming token embeddings for both the input projection and gate
        inx, gatex = token_shift(x, x_state, self.Wlerp_in, self.Wlerp_g)

        # project to 3x larger hidden dimension
        # this is 4x for vanilla transformers FFN, but it's typical to reduce it when adding new parameters 
        #  to allow comparison models with the same number of total parameters for a given d_model, n_layer
        # in rwkv5's case, if you drop it down to 3x you end up making up for the extra parameters in the gate
        hidden = self.Win(inx)

        # relu^2 activation function
        hidden = torch.square(torch.relu(hidden))

        # project back out to d_model
        out = self.Wout(hidden)

        # apply sigmoid gate
        gate = self.Wgate(gatex)
        gate = torch.sigmoid(gate)
        out = out * torch.sigmoid(gate)

        return x + out, x[:, -1]


if __name__ == "__main__":
    L,B,T,C,H,K,V = 12, 1, 2, 768, 12, 64, 64
    model = RWKV5()
    model.forward(torch.ones(B,T,dtype=torch.long))
