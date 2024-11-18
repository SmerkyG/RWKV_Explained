import math
import torch
from torch import nn, Tensor

# config is unchanged from RWKV5
from rwkv5_2 import Config

def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            assert False
        return x

class LayerState:
    # the recurrent neural network (RNN) state for a layer of RWKV7
    def __init__(self, x, cfg:Config):
        B, T, C, H, K = x.size(0), x.size(1), cfg.d_model, cfg.n_heads, cfg.d_model // cfg.n_heads
        V = K
        # a (B,C) size tensor representing latest time mixer token embedding processed
        self.time_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)
        # an (B,H,V,K) size tensor representing a decaying token embedding memory for each head, where H=number_of_heads, K=key_dim_per_head, V=value_dim_per_head 
        self.vk_state = torch.zeros(B,H,V,K,dtype=x.dtype,device=x.device)
        # a (B,C) size tensor representing latest channel mixer token embedding processed
        self.channel_mixer_x_state = torch.zeros(B,C,dtype=x.dtype,device=x.device)

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

        # run each layer in succession, passing in the 'value' originally calculated in layer zero and RNN state for that layer
        v0 = torch.tensor([], device=x.device, dtype=x.dtype)
        for layer_id, block in enumerate(self.layers):  # run each rwkv block
            x, v0, s[layer_id] = block(x, v0, s[layer_id])

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

    def forward(self, x : Tensor, v0 : Tensor, s : LayerState):
        # PLEASE NOTE that the vk_state is in HVK order, *not* HKV as it was in RWKV-5 and RWKV-6, when it was called kv_state!
        x, v0, s.time_mixer_x_state, s.vk_state = self.time_mixer(x, v0, s.time_mixer_x_state, s.vk_state)
        x, s.channel_mixer_x_state = self.channel_mixer(x, s.channel_mixer_x_state)
        return x, v0, s

class LoRA_Simple(nn.Module):
    def __init__(self, dim:int, dim_hidden:int, init_value : Tensor|None = None):
        super().__init__()
        init_value = init_value if init_value is not None else torch.zeros(dim)
        self.base = nn.Parameter(init_value)
        self.W_a = nn.Linear(dim, dim_hidden, bias=False)
        self.W_b = nn.Linear(dim_hidden, dim, bias=False)

        with torch.no_grad():
            self.W_a.weight.zero_()
            ortho_init(self.W_b.weight, 0.1)

    def forward(self, x : Tensor): # x (B,T,C)
        # this is rwkv's version of low rank adaptation

        # the result has two components: a base value vector, and an offset
        # the offset is calculated by taking token shifted x and squeezing it through shrinking and expanding linear layers
        # this offers greatly reduced cost in terms of both computation and parameters than a single dim->dim linear layer
        return self.base + self.W_b( self.W_a(x) )

class LoRA_MLP(LoRA_Simple):
    def forward(self, x : Tensor): # x (B,T,C)
        # this is rwkv's version of low rank adaptation, with an added activation in the middle

        # the result has two components: a base value vector, and an offset
        # the offset is calculated by taking token shifted x and squeezing it through shrinking and expanding linear layers
        # using tanh as an activation in the middle of that sandwich
        # this offers greatly reduced cost in terms of both computation and parameters than a single dim->dim linear layer
        return self.base + self.W_b( nn.functional.tanh( self.W_a(x) ) )

class TimeMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg

        d_attn = d_model = cfg.d_model

        with torch.no_grad():
            self.time_mixer_prenorm = nn.LayerNorm(cfg.d_model)

            self.lerps = [nn.Linear(d_model, d_model, bias=False) for _ in range(6)]
            self.decay_lora = LoRA_MLP(d_model, 64, torch.ones(d_model))
            self.W_proj_r = nn.Linear(d_model, d_model, bias=False)
            self.W_proj_k = nn.Linear(d_model, d_model, bias=False)
            self.W_proj_v = nn.Linear(d_model, d_model, bias=False)
            self.W_proj_out = nn.Linear(d_model, d_model, bias=False)
            
            self.decay_lora_mlp = LoRA_MLP(d_model, 64 if d_model < 4096 else 128) # dim 64 for emb 768, change it for smaller/larger models
           
            self.iclr_lora = LoRA_Simple(d_model, 32) # dim 32 for emb 768, change it for smaller/larger models

            self.W_deformed_key = nn.Parameter(torch.ones(1, 1, d_model))
            
            D_GATE_LORA = 128 # dim 128 for emb 768, change it for smaller/larger models
            self.gate_w1 = nn.Parameter(torch.zeros(d_model, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(torch.zeros(D_GATE_LORA, d_attn).uniform_(-0.01, 0.01))
            ortho_init(self.gate_w2, 0.1)

            self.iclr_mix_amt = nn.Parameter(torch.ones(1, 1, d_model))

            self.v0_mix_amt_lora = LoRA_Simple(d_model, 64) # dim 64 for emb 768, change it for smaller/larger models

            # per-channel boost for current embedding
            self.u = nn.Parameter(torch.ones(cfg.n_heads, cfg.d_model//cfg.n_heads))

            self.group_norm = nn.GroupNorm(cfg.n_heads, cfg.d_model, eps=64e-5)

    def _init_weights(self, module):            
        d_model = self.cfg.d_model
        if isinstance(module, nn.Linear):
            if module == self.W_proj_r or module == self.W_proj_v:
                module.weight.data.uniform_(-0.5/(d_model**0.5), 0.5/(d_model**0.5))
            elif module == self.W_proj_k:
                module.weight.data.uniform_(-0.05/(d_model**0.5), 0.05/(d_model**0.5))
            elif module == self.W_proj_out:
                module.weight.data.zero_()

    # unused, but shows how this can be accomplished with a state transition matrix
    @staticmethod
    def single_timestep_transition_matrix(r, k, v, transition_matrix, vk_state): 
        # transform inputs from BHK into column vectors BHK1
        r, k, v = map(lambda x: x.unsqueeze(-1), (r, k, v))

        # decay the kv state
        vk_state = vk_state @ transition_matrix # BHVK @ BHVK = BHVK

        # add in an dynamically iclr mixed amount of the latest value at the key 
        # (key has been pre-adjusted in the calling code by the amount of iclr mixing)
        vk_state = vk_state + (v.mT @ k)   # BHVK

        # apply receptance to the new state
        out = vk_state @ r  # BHVK @ BHK1 = BHV1

        # remove an extra useless dimension from the output
        return out.squeeze(-1), vk_state # BHV, BHVK

    @staticmethod
    def single_timestep(r, k, v, decay, iclr, deformed_key, vk_state): 
        # PLEASE NOTE that the vk_state is in HVK order, *not* HKV as it was in RWKV-5 and RWKV-6, when it was called kv_state!

        # transform inputs from BHK into column vectors BHK1
        r, k, v, decay, iclr, deformed_key = map(lambda x: x.unsqueeze(-1), (r, k, v, decay, iclr, deformed_key))

        # decay the kv state
        vk_state = vk_state * decay.mT # BHVK * BH1K = BHVK

        # remove the iclr amount of the value stored within the state at the deformed key
        vk_state = vk_state - vk_state @ deformed_key @ (iclr * deformed_key).mT

        # add in an dynamically iclr and 1-decay mixed amount of the latest value at the key 
        # (key has been pre-adjusted in the calling code by the amount of iclr mixing)
        vk_state = vk_state + (v.mT @ k)   # BHVK

        # apply receptance to the new state
        out = vk_state @ r  # BHVK @ BHK1 = BHV1

        # remove an extra useless dimension from the output
        return out.squeeze(-1), vk_state # BHV, BHVK

    def forward(self, x : Tensor, v0 : Tensor, x_state : Tensor, vk_state : Tensor): # x (B,T,C), x_state (B,C), vk_state (B,H,V,K)
        # PLEASE NOTE that the vk_state is in HVK order, *not* HKV as it was in RWKV-5 and RWKV-6, when it was called kv_state!

        B, T, C, H, K = x.size(0), x.size(1), self.cfg.d_model, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads

        x = self.time_mixer_prenorm(x)

        # we want the token embeddings shifted over by one towards the past
        # to get this, we take the last token embedding processed and append all but one of the current token embeddings to it
        # (the last token embedding processed is what's stored in the x_state)
        x_shifted_one_to_the_past = torch.cat((x_state.unsqueeze(-2), x[:,:-1]), dim=1)

        # token shift the incoming token embeddings for the receptance, key, value, decay, iclr, and gate
        # token shift is just a learned linear interpolation between the current and previous token embeddings in the sequence
        # this is done by lerping between x and the shifted x we just calculated
        # note that it is a per-channel learned interpolation amount, not just a single value per head
        x_decay, x_key, x_value, x_receptance, x_iclr, x_gate = [torch.lerp(x, x_shifted_one_to_the_past, lerp_amt_weights(x)) for lerp_amt_weights in self.lerps]
                                           
        # project and separate out our vectors into attention heads
        # the extra dimensions are being added here to enable matrix multiplications per timestep
        r = self.W_proj_r(x_receptance) # BTC
        k = self.W_proj_k(x_key)        # BTC
        v = self.W_proj_v(x_value)      # BTC

        # dynamically interpoate values between the value from the first layer (v0) and the value for the current layer (v)
        if self.layer_id == 0:
            # in the first layer, just return the value for use in later layers rather than interpolate
            v0 = v
        else:
            v = torch.lerp(v, v0, torch.sigmoid(self.v0_mix_amt_lora(x_value)))

        # gate is generated using a LoRA style low parameter method with no base
        gate = torch.sigmoid(x_gate @ self.gate_w1) @ self.gate_w2 # BTC

        # decay is generated using a LoRA-MLP low parameter method, and then soft clamped to the range [-inf, -0.5]
        log_neglog_of_decay = self.decay_lora_mlp(x_decay)                         # BTC
        log_neglog_of_decay = -0.5 - nn.functional.softplus(-log_neglog_of_decay)   # BTC
        log_of_decay = -torch.exp(log_neglog_of_decay)
        decay = log_of_decay.exp()

        # the next section is hard to understand unless you first understand how RWKV-7 modifies the delta rule:

        # The traditional delta rule removes some amount (the 'in-context learning rate' iclr) of the old value 
        #  stored in the state at the current key and replaces that same amount with the new value at the current key.
        #  This is accomplished by 
        #  1) projecting the state onto the current key: (state @ k @ k.T)
        #  2) multiplying by the in-context learning rate: iclr * (state @ k @ k.T)
        #  3) subtracting the result off of the state: state - iclr * (state @ k @ k.T)
        #  4) adding the iclr amount of the new value to the state: state - iclr * (state @ k @ k.T) + iclr * (v.T @ k)
        #  Therefore, we remove iclr amount of the 'old value' contained within the state at the current key and replace it with iclr amount of the 'new value' at the current key

        # RWKV-7 reconceptualizes this rule, such that we:
        #  1) decay the current state
        #  2) remove an iclr amount of the value currently stored in the state at the *deformed key*
        #  3) add in a *varying* amount of the new value at the current key, based on the iclr
        # This can be done via the formula: state = state * decay - state @ deformed_key @ (iclr * deformed_key.T) + v.T @ (adjusted_iclr * k)
        # steps 1 and 2 [state = state * decay - state @ deformed_key @ (iclr * deformed_key.T)] can be combined into a single state transition matrix per timestep, 
        #  which can be multiplied by the state to obtain the next state

        # the deformed key is used as the modified key to remove during the delta-rule portion of the kernel

        # the deformed key is generated using a LoRA style low parameter method with the original key as the base, and then normalized
        deformed_key = k * self.W_deformed_key
        deformed_key = nn.functional.normalize(deformed_key.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)

        # iclr ('in-context learning rate') is generated using a LoRA style low parameter method
        iclr = torch.sigmoid( self.iclr_lora(x_iclr) )

        # the state transition matrix (see above) - not used, just for descriptive purposes
        #state_transition_matrix = torch.diag(decay) - deformed_key @ (iclr * deformed_key).mT

        # the varying amount of the new value added is determined by a dynamic mix of the in-context learning rate
        # dynamically interpolate keys between original key and key*iclr (this is for step 3 above)
        k = torch.lerp(k, k * iclr, self.iclr_mix_amt)

        # separate into heads (B,T,H,K)
        r, k, v, decay, iclr, deformed_key = map(lambda x: x.view(B,T,H,-1), (r, k, v, decay, iclr, deformed_key))

        out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
        for t in range(T):
            out[:,t], vk_state = TimeMixer.single_timestep(r[:,t], k[:,t], v[:,t], decay[:,t], iclr[:,t], deformed_key[:,t], vk_state)

        # apply group normalization to each head and recombine the heads
        out = self.group_norm(out.view(B*T, C)).view(B, T, C) # BTC

        # add in the bonus term
        bonus = ((r*k*self.u).sum(dim=-1, keepdim=True) * v)
        bonus = bonus.view(B,T,C)   # recombine bonus heads
        out = out + bonus

        # apply gate to the output
        out = out * gate # BTC

        # project the output
        out = self.W_proj_out(out) # BTC

        return x + out, v0, x[:, -1], vk_state

class ChannelMixer(nn.Module):
    def __init__(self, cfg:Config, layer_id:int):
        super().__init__()
        self.cfg = cfg
        self.prenorm = nn.LayerNorm(cfg.d_model)
        self.W_in = nn.Linear(cfg.d_model, cfg.d_model * 4, bias=False)
        self.W_out = nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False)
        self.W_lerp_in = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x : Tensor, x_state : Tensor): # x (B,T,C), x_state (B,C)
        x = self.prenorm(x)

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
        inx = torch.lerp(x, x_shifted_one_to_the_past, self.W_lerp_in(x))

        # project to 4x larger hidden dimension
        hidden = self.W_in(inx)

        # relu^2 activation function
        hidden = torch.square(torch.relu(hidden))

        # project back out to d_model
        out = self.W_out(hidden)

        return x + out, x[:, -1]
    
if __name__ == "__main__":
    model = RWKV()
    model.forward(torch.ones(1,2,dtype=torch.long))
