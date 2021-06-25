# Transformer baselines

Transformer baselines should be run with official fairseq. Please copy the scripts to fairseq root folder.

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


