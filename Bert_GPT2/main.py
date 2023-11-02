from datasets import load_dataset
from datasets import load_metric # use evaluate instead in the future version
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# args
parser = argparse.ArgumentParser(
    description="Fine-tune All kinds of Transformer on IMDB for sequence classification"
)
parser.add_argument("--model_name", type=str, default="bart", help="transformer model name")

parser.add_argument('--eval', default=False, action='store_true', help='only evaluate the model')
parser.add_argument('--bs', default=1, type=int, help='batch size')


args = parser.parse_args()


imdb = load_dataset("imdb")


if args.model_name == "gpt2":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
elif args.model_name == "bert":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
elif args.model_name == "bart":
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_imdb = imdb.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


metric = load_metric("accuracy")




def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)



id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# mytransformers is the huggingface transformers library modified by adding column permutations.
if args.model_name == "gpt2":
    from mytransformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2, id2label=id2label, label2id=label2id
    )
    state_dict = torch.load('./imdb-gpt2/encrypted_ori_model.bin')
    model.load_state_dict(state_dict)
    
elif args.model_name == "bert":
    from mytransformers.models.bert.modeling_bert import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
    torch.backends.cudnn.enable =True
    torch.backends.cudnn.benchmark = True
    state_dict = torch.load('./imdb-bert/encrypted_ori_model.bin')
    model.load_state_dict(state_dict)

bs = args.bs
device = "cpu" # for debug
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
if not args.eval:
    trainer.train() 
# 
else: 
    state_dict = torch.load('./imdb-gpt2/encrypted_pytorch_model.bin')
    model.load_state_dict(state_dict)
results = trainer.evaluate()
print(results)

trainer.save_model("./imdb-" + args.model_name)

