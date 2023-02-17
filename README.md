# Translationese

## Getting Started
To install the dependencies: `pip3 install -r requirements.txt`

To download the data: `bash scripts/download_data.sh`

## Reproducing
- PAWS-X `bash scripts/pawsx.sh` 

- XNLI `bash scripts/xnli.sh`

### XLM-R Large Results 
| Setting | XNLI | PAWSX |
| ------- | ---- | ----- |
| [Reported](https://arxiv.org/abs/1911.02116)| 79.2 | 86.4 |
| Reproduced |  | 87.5 |
| [SAGE](https://openreview.net/pdf?id=cuvga_CiVND) |      | 87.0 |
| [R3F](https://arxiv.org/abs/2008.03156)  |      | **88.1** |
| [EWC](https://arxiv.org/abs/1612.00796)  |      | 87.8|
| [Dropconnect](https://proceedings.mlr.press/v28/wan13.html) | | **88.1** |
### Running with your own configuration of hyper-parameters

First see `utils.py` and add your task, then run

```
python3 finetune_xlmr.py --task {your task} \
--lr {learning rate} --bs {batch size} --regularizer {ewc or r3f} --epoch {num epochs}
```

The arguments should be self explainatory.
