from pydoc import locate
import torch, sys, os
from torch.nn import functional as F
import yaml
from transformers import AutoTokenizer

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 inference.py config_path model_path")
        exit(-1)

    cfg_path = sys.argv[1]
    model_path = sys.argv[2]
    with open(cfg_path, mode="rt", encoding="utf-8") as file:
        config = yaml.load(file, yaml.SafeLoader)

    module_path = config['module_path']
   
    rwkv_module = locate(module_path)
    if rwkv_module is None:
        print(f"No rwkv module found at {module_path}")
        exit(-1)

    RWKV = rwkv_module.RWKV
    Config = rwkv_module.Config
    convert_params_from_pth = rwkv_module.convert_params_from_pth

    model_config = Config(**config['model'])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

    DTYPE = torch.bfloat16
    #DTYPE = torch.half # better for RWKV-7

    # model download: https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main
    model_params = torch.load(model_path, weights_only=True)
    model_params = convert_params_from_pth(model_params)

    with torch.no_grad():
        model = RWKV(model_config).to(dtype=DTYPE).cuda()
        model.load_state_dict(model_params)

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt)
    print(f'\nInput:\n{input}')

    out, state = model.forward(torch.tensor(input).reshape(1,-1).cuda())
    print(f'\nOutput:\n{out}')

    # let's check the logits for the last token => prediction for the next token    
    out = out[0, -1]
    
    probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

    print(f'\n{prompt}')

    _, indices = torch.topk(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')

if __name__ == "__main__":
    main()
