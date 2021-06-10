# G-Transformer

**This code is for ACL 2021 paper [G-Transformer for Document-level Machine Translation](https://arxiv.org/abs/2105.14761).**

**Python Version**: Python3.6

**Package Requirements**: torch==1.4.0 tensorboardX

**Framework**: Our model and experiments are built upon [fairseq](https://github.com/pytorch/fairseq).

Before running the scripts, please install fairseq dependencies by:
```
    pip install --editable .
```

## Non-pretraining Settings

### G-Transformer random initialized
* Prepare data: 
```
    mkdir exp_randinit
    bash exp_gtrans/run-all.sh prepare-randinit exp_randinit
```

* Train model:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp_gtrans/run-all.sh run-randinit train exp_randinit
```

* Evaluate model:
```
    bash exp_gtrans/run-all.sh run-randinit test exp_randinit
```

### G-Transformer fine-tuned on sent Transformer
* Prepare data: 
```
    mkdir exp_finetune
    bash exp_gtrans/run-all.sh prepare-finetune exp_finetune
```

* Train model:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp_gtrans/run-all.sh run-finetune train exp_finetune
```

* Evaluate model:
```
    bash exp_gtrans/run-all.sh run-randinit test exp_finetune
```

## Pretraining Settings
* Prepare data: 
```
    mkdir exp_mbart
    bash exp_gtrans/run-all.sh prepare-mbart exp_mbart
```

* Train model:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp_gtrans/run-all.sh run-mbart train exp_mbart
```

* Evaluate model:
```
    bash exp_gtrans/run-all.sh run-mbart test exp_mbart
```
