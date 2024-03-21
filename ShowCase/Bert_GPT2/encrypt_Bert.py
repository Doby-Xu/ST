import torch
from transformers import BertForSequenceClassification

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
ori_state_dict = model.state_dict()
# print(ori_state_dict.keys())
# Get the state_dict with "bert.encoder.layer." as prefix
new_state_dict = {}
for k, v in ori_state_dict.items():
    if k.startswith("bert.encoder.layer."):
        new_state_dict[k[19:]] = v

# Get the parameters and names in the new state dict 

params = []
names = []
for k, v in new_state_dict.items():
    params.append(v)
    names.append(k)
    print(k, v.shape)

# get keys
pc, ipc = torch.load('keys/key_768_m.pt'), torch.load('keys/unkey_768_m.pt')
# to cuda
# pc, ipc = pc.cuda(), ipc.cuda()

# traverse the names and params, find those with "qkv.weight" in it
for i in range(len(names)):
    
    if "attention" in names[i]:
        para = params[i]
        # B P_C^-1
        para = torch.matmul(para, ipc)
        if "weight" in names[i] and "LayerNorm" not in names[i]:
            # P_C W
            para = torch.matmul(pc, para)
        params[i] = para
        print(names[i], "with shape" , params[i].shape, "is encrypted with P_C W P_C^-1" if "weight" in names[i] else "is encrypted with B P_C^-1")
    elif "intermediate" in names[i]:
        if "weight" in names[i]:
            weight = params[i]
            # W P_C^-1
            weight = torch.matmul(weight, ipc)
            params[i] = weight
            print(names[i], "with shape" , params[i].shape, "is encrypted by W P_C^-1")
        elif "bias" in names[i]:
            print(names[i], "with shape" , params[i].shape, "should not be encrypted")
        else:
            raise ValueError("Error: " + names[i] + "should not be here")
    elif "output" in names[i]:
        para = params[i]
        if "weight" in names[i]:
            # P_C W
            para = torch.matmul(pc, para)
        else:
            # B P_C^-1
            para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape" , params[i].shape, "is encrypted with P_C W" if "weight" in names[i] else "is encrypted with B P_C^-1")

    else:
        raise ValueError("Error: " + names[i] + "should not be here")
    
# add the prefix "backbone.blocks" to the names, and replace the corresponding params in the original state_dict, save it
for i in range(len(names)):
    names[i] = "bert.encoder.layer." + names[i]
    ori_state_dict[names[i]] = params[i]
    print(names[i], "is done")
torch.save(ori_state_dict, './imdb-bert/encrypted_ori_model.bin')



