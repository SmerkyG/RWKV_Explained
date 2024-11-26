from pydoc import locate
import torch, sys
from torch.nn import functional as F

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage:")
        print("  python3 inference.py [model_path] [rwkv_module_path]")
        exit(-1)

    rwkv_module_path = sys.argv[2] if len(sys.argv) >= 3 else 'rwkv7'
    model_path = sys.argv[1]
   
    rwkv_module = locate(rwkv_module_path)
    if rwkv_module is None:
        print(f"No rwkv module found at {rwkv_module_path}")
        exit(-1)

    RWKV = rwkv_module.RWKV
    Config = rwkv_module.Config
    convert_params_from_pth = rwkv_module.convert_params_from_pth

    config = Config()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("20B_tokenizer.json")

    # DTYPE = torch.bfloat16
    DTYPE = torch.half # better

    # model download: https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main
    model_params = torch.load(model_path, weights_only=True)
    model_params = convert_params_from_pth(model_params)

    with torch.no_grad():
        model = RWKV(config).to(dtype=DTYPE).cuda()
        model.load_state_dict(model_params)

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt).ids
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
