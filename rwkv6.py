import torch
from torch import nn, Tensor
from dataclasses import dataclass

# config, layer state and channel mixer are unchanged from RWKV5
from rwkv5_2 import Config, LayerState

# Only time mixer changes for RWKV6
class RWKV(torch.nn.Module):
    def __init__(self, cfg:Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
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
    
class LoRA_MLP(nn.Module):
    def __init__(self, dim:int, dim_hidden:int, init_value : Tensor|None = None):
        super().__init__()
        init_value = init_value if init_value is not None else torch.empty(1, 1, dim)
        self.base = nn.Parameter(init_value)
        self.W_a = nn.Parameter(torch.empty(dim, dim_hidden))
        self.W_b = nn.Parameter(torch.empty(dim_hidden, dim))

    def forward(self, x : Tensor): # x (B,T,C)
        # this is rwkv's version of low rank adaptation

        # the result has two components: a base value vector, and an offset
        # the offset is calculated by taking token shifted x and squeezing it through shrinking and expanding linear layers
        # using tanh as an activation in the middle of that sandwich
        # this offers greatly reduced cost in terms of both computation and parameters than a single dim->dim linear layer
        return self.base + nn.functional.tanh( x @ self.W_a ) @ self.W_b

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

@dataclass
class LoRARanks:
    min_d_model:int
    ddlerp_lora:int
    decay_lora:int

class TimeMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.cfg = cfg

        d_model = cfg.d_model
        d_head = d_model // cfg.n_heads

        self.prenorm = nn.LayerNorm(d_model)

        lora_ranks_by_dim = [
            LoRARanks(min_d_model=0,    decay_lora=64,  ddlerp_lora=32),
            LoRARanks(min_d_model=4096, decay_lora=128, ddlerp_lora=64),
        ]
        # find lora ranks for current d_model
        for lora_ranks_iter in lora_ranks_by_dim:
            if lora_ranks_iter.min_d_model > d_model:
                break
            lora_ranks = lora_ranks_iter

        self.ddlerp_premix = nn.Parameter(torch.empty(1, 1, d_model))
        self.ddlerp_decay = DDLerp(d_model, lora_ranks.ddlerp_lora)
        self.ddlerp_key = DDLerp(d_model, lora_ranks.ddlerp_lora)
        self.ddlerp_value = DDLerp(d_model, lora_ranks.ddlerp_lora)
        self.ddlerp_receptance = DDLerp(d_model, lora_ranks.ddlerp_lora)
        self.ddlerp_gate = DDLerp(d_model, lora_ranks.ddlerp_lora)

        self.decay = LoRA_MLP(d_model, lora_ranks.decay_lora)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # per-channel boost for current embedding
        self.bonus = nn.Parameter(torch.ones(cfg.n_heads, d_head))

        self.group_norm = nn.GroupNorm(cfg.n_heads, d_model, eps=64e-5)

    @staticmethod
    def single_timestep(r, k, v, u, w, kv_state): # all BHK except kv_state which is BHKV
        original_dtype = r.dtype

        B, H, K, V = kv_state.shape

        # transform inputs from BHK and put everything in float format for higher precision
        r = r.float().view(B, H, 1, K)
        w = w.float().view(B, H, K, 1)
        k = k.float().view(B, H, K, 1)
        v = v.float().view(B, H, 1, V)
        u = u.float().view(1, H, K, 1)

        kv = k @ v # BHK1 @ BH1V = BHKV

        # start with the existing kv state
        y = kv_state        # BHKV
        # apply the u boost to the current k @ v and add it to that
        y = y + kv * u # BHKV + BHKV * 1HK1 = BHKV
        # apply receptance to that whole result
        out = r @ y         # BH1K @ BHKV = BH1V
        # remove an extra useless dimension from the output
        out = out.squeeze(-2).to(original_dtype) # BHV

        # finally, decay the kv state and add in the latest k @ v
        kv_state = kv_state * w    # BHKV * BHK1 = BHKV
        kv_state = kv_state + kv   # BHKV + BHKV = BHKV

        return out, kv_state # BHV, BHKV

    def forward(self, hidden_state_in : Tensor, x_state : Tensor, kv_state : Tensor): # x (B,T,C), x_state (B,C), kv_state (B,H,K,V)
        x = self.prenorm(hidden_state_in)
        x_state_out = x[:, -1]

        B, T, C, H, K = x.size(0), x.size(1), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads

        # we want the token embeddings shifted over by one towards the past
        # to get this, we take the last token embedding processed and append all but one of the current token embeddings to it
        # (the last token embedding processed is what's stored in the x_state)
        x_shifted_one_to_the_past = torch.cat((x_state.unsqueeze(-2), x[:,:-1]), dim=1)

        # token shift the incoming token embeddings for the receptance, key, value, gate, and decay
        # PLEASE NOTE THAT THE DIRECTION OF THE LERP CHANGED IN RWKV-6
        x_premixed = torch.lerp(x, x_shifted_one_to_the_past, self.ddlerp_premix)
        x_decay = self.ddlerp_decay(x_premixed, x, x_shifted_one_to_the_past)
        x_key = self.ddlerp_key(x_premixed, x, x_shifted_one_to_the_past)
        x_value = self.ddlerp_value(x_premixed, x, x_shifted_one_to_the_past)
        x_receptance = self.ddlerp_receptance(x_premixed, x, x_shifted_one_to_the_past)
        x_gate = self.ddlerp_gate(x_premixed, x, x_shifted_one_to_the_past)
                                           
        # project and separate out our vectors into attention heads
        r = self.receptance(x_receptance).view(B,T,H,K) # BTHK
        k = self.key(x_key).view(B,T,H,K) # BTHK
        v = self.value(x_value).view(B,T,H,K) # BTHK
        gate = self.gate(x_gate) # BTC

        # per-channel data-dependent decays generated inexpensively via low rank adaptation
        log_neglog_of_decay = self.decay(x_decay)
        # separate out into attention heads
        log_neglog_of_decay = log_neglog_of_decay.view(B,T,H,K)
        # this forces the decays to end up in the range 0...1 using a nicely differentiable function
        log_of_decay = -torch.exp(log_neglog_of_decay.float())
        decay = log_of_decay.exp() # BTHK

        out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
        for t in range(T):
            out[:,t], kv_state = TimeMixer.single_timestep(r[:,t], k[:,t], v[:,t], self.bonus, decay[:,t], kv_state)

        # apply group normalization to each head and recombine the heads
        out = self.group_norm(out.view(B*T, C)).view(B, T, C) # BTC

        # apply silu gate to the output
        out = out * nn.functional.silu(gate) # BTC

        # project the output
        out = self.output(out) # BTC

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
        # PLEASE NOTE THAT THE DIRECTION OF THE LERP CHANGED IN RWKV-6
        x_in = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_in)
        x_gate = torch.lerp(x, x_shifted_one_to_the_past, self.tokenshift_gate)

        # project to 3.5x larger hidden dimension
        # this is 4x for vanilla transformers FFN, but it's typical to reduce it when adding new parameters 
        #  to allow comparison models with the same number of total parameters for a given d_model, n_layer
        # if you drop it down to 3.5x you end up making up for the extra parameters in the gate
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

def convert_params_from_pth(model_params):   
    # map the state dict entries to our naming convention

    model_params['embed'] = model_params.pop('emb.weight')
    model_params['embed_norm.weight'] = model_params.pop('blocks.0.ln0.weight')
    model_params['embed_norm.bias'] = model_params.pop('blocks.0.ln0.bias')

    model_params['lm_head_norm.weight'] = model_params.pop('ln_out.weight')
    model_params['lm_head_norm.bias'] = model_params.pop('ln_out.bias')
    model_params['lm_head_unembed.weight'] = model_params.pop('head.weight')

    replacements = {
        '.ln1.weight' : '.time_mixer.prenorm.weight',
        '.ln1.bias' : '.time_mixer.prenorm.bias',

        '.ln2.weight' : '.channel_mixer.prenorm.weight',
        '.ln2.bias' : '.channel_mixer.prenorm.bias',

        # time mixer token shift lerp bases
        '.att.time_maa_x' : '.time_mixer.ddlerp_premix',
        '.att.time_maa_r' : '.time_mixer.ddlerp_receptance.lora.base',
        '.att.time_maa_w' : '.time_mixer.ddlerp_decay.lora.base',
        '.att.time_maa_k' : '.time_mixer.ddlerp_key.lora.base',
        '.att.time_maa_v' : '.time_mixer.ddlerp_value.lora.base',
        '.att.time_maa_g' : '.time_mixer.ddlerp_gate.lora.base',

        # bonus
        '.time_faaaa' : '.bonus',

        # decay_lora
        '.time_decay' : '.decay.base',
        '.time_decay_w1' : '.decay.W_a',
        '.time_decay_w2' : '.decay.W_b',

        # group_norm
        '.ln_x.weight' : '.group_norm.weight',
        '.ln_x.bias' : '.group_norm.bias',

        # channel mixer
        '.ffn.time_maa_k' : '.channel_mixer.tokenshift_in',
        '.ffn.time_maa_r' : '.channel_mixer.tokenshift_gate',
        '.ffn.key.weight' : '.channel_mixer.W_in.weight',
        '.ffn.value.weight' : '.channel_mixer.W_out.weight',
        '.ffn.receptance.weight' : '.channel_mixer.gate.weight',
    }

    for k in list(model_params.keys()):
        p = model_params.pop(k)

        for needle, replacement in replacements.items():
            if k.endswith(needle):
                k = k.replace(needle, replacement)

        k = k.replace('blocks.', 'layers.')

        k = k.replace('.att.', '.time_mixer.')
        k = k.replace('.ffn.', '.channel_mixer.')

        maa_names = ['decay', 'key', 'value', 'receptance', 'gate']
        if k.endswith('.time_mixer.time_maa_w1'):
            for i, w in enumerate(p.chunk(5, dim=1)):
                model_params[k.replace('.time_maa_w1', f'.ddlerp_{maa_names[i]}.lora.W_a')] = w
            continue

        if k.endswith('.time_mixer.time_maa_w2'):
            for i, w in enumerate(p.chunk(5, dim=0)):
                model_params[k.replace('.time_maa_w2', f'.ddlerp_{maa_names[i]}.lora.W_b')] = w.squeeze(0)
            continue

        model_params[k] = p

    return model_params
