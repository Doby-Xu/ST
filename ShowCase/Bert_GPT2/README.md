## Try permutation properties with BERT and GPT2

Here we provide simple code for validating the properties with BERT and GPT2.

First, you need to download the huggingface transformers library, and put the `src/transformers` in `mytransformers` folder, but keep our `modeling_bert.py` and `modeling_gpt2.py` in this repository.

Then you can try to encrypt the pre-trained BERT and GPT2 models with matrix multiplication permutation.

```bash
python encrypt_bert.py
```

And then you can try to validate the column permutation property by training the model with/without column permutation.
    
```bash
python main.py
```

Column shuffling is on by default, you can turn it off by commenting out the matrix multiplication in `modeling_bert.py` and `modeling_gpt2.py`. 

```python

    # shuffle
    self.pc, self.ipc = self.pc.to(device), self.ipc.to(device)
    embedding_output = torch.matmul(embedding_output, self.ipc) # comment out this line to turn off column permutation

    encoder_outputs = self.encoder(
        ...
    )
    
    sequence_output = encoder_outputs[0]

    # shuffle
    sequence_output = torch.matmul(sequence_output, self.pc)# comment out this line to turn off column de-permutation

```



