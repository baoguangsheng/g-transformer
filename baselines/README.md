# Transformer baseline

Transformer baselines should be run with official fairseq. 

* Prepare code:
Please copy the scripts to fairseq root folder.

Appending model setting into fairseq/models/transformer.py
```
@register_model_architecture("transformer", "transformer_base")
def transformer_doc_base(args):
	args.encoder_layers = getattr(args, "encoder_layers", 6)
	args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
	args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
	args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
	args.decoder_layers = getattr(args, "decoder_layers", 6)
	args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
	args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
	args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
	args.dropout = getattr(args, "dropout", 0.3)
	base_architecture(args)

```

* Prepare data: 
Follow the readmes under folder raw_data to prepare the raw data first, then
```
mkdir exp_sent
bash run-sent.sh data exp_sent iwslt17
bash run-sent.sh data exp_sent nc2016
bash run-sent.sh data exp_sent europarl7
```

* Train model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run-sent.sh train exp_sent iwslt17
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run-sent.sh train exp_sent nc2016
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run-sent.sh train exp_sent europarl7
```

* Evaluate model:
```
bash run-sent.sh test exp_sent iwslt17
bash run-sent.sh test exp_sent nc2016
bash run-sent.sh test exp_sent europarl7
```


