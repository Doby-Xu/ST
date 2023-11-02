import torch
from einops import rearrange
from transformers import GPT2ForSequenceClassification

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2, id2label=id2label, label2id=label2id
    )
ori_state_dict = model.state_dict()
# print(ori_state_dict.keys())

# Get the state_dict with "transformer.h." as prefix
new_state_dict = {}
for k, v in ori_state_dict.items():
    if k.startswith("transformer.h."):
        new_state_dict[k[14:]] = v

# Get the parameters and names in the new state dict
params = []
names = []
for k, v in new_state_dict.items():
    params.append(v)
    names.append(k)
    print(k, v.shape)

# get keys
pc, ipc = torch.load('keys/key_768_m.pt'), torch.load('keys/unkey_768_m.pt')

# traverse the names and params, find those with "qkv.weight" in it
for i in range(len(names)):
    if "attn.c_attn.weight" in names[i]:
        weight = params[i]
        weight = rearrange(weight, 'd (h w) -> h d w', h = 3)
        weight = torch.matmul(pc, weight)
        weight = torch.matmul(weight, ipc)
        weight = rearrange(weight, 'h d w -> d (h w)')
        params[i] = weight
        print(names[i], "with shape" , params[i].shape, "is encrypted by P_C W P_C^-1")
    elif "attn.c_attn.bias" in names[i]:
        bias = params[i]
        bias = rearrange(bias, '(h d) -> h d', h = 3)
        bias = torch.matmul(bias, ipc)
        bias = rearrange(bias, 'h d -> (h d)')
        params[i] = bias
        print(names[i], "with shape" , params[i].shape, "is encrypted by B P_C^-1")
    elif "ln" in names[i] or "attn.c_proj.bias" in names[i] or "mlp.c_proj.bias" in names[i]:
        bias = params[i]
        bias = torch.matmul(bias, ipc)
        params[i] = bias
        print(names[i], "with shape" , params[i].shape, "is encrypted by B P_C^-1")
    elif "attn.c_proj.weight" in names[i]:
        weight = params[i]
        weight = torch.matmul(pc, weight)
        weight = torch.matmul(weight, ipc)
        params[i] = weight
        print(names[i], "with shape" , params[i].shape, "is encrypted by P_C W P_C^-1")
    elif "mlp.c_fc.weight" in names[i]:
        weight = params[i]
        weight = torch.matmul(pc, weight)
        params[i] = weight
        print(names[i], "with shape" , params[i].shape, "is encrypted by P_C W")
    elif "mlp.c_fc.bias" in names[i]:
        pass # do nothing
    elif "mlp.c_proj.weight" in names[i]:
        
        weight = params[i]
        # weight = torch.matmul(pc, weight)
        weight = torch.matmul(weight, ipc)
        params[i] = weight
        print(names[i], "with shape" , params[i].shape, "is encrypted by W P_C^-1")
    else:
        raise ValueError("Error: " + names[i] + "should not be here")
    
# add the prefix "transformer.h." to the names, and replace the corresponding params in the original state_dict, save it
for i in range(len(names)):
    names[i] = "transformer.h." + names[i]
    ori_state_dict[names[i]] = params[i]

torch.save(ori_state_dict, './imdb-gpt2/encrypted_ori_model.bin')
    
