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
| Reproduced |  ||
| [SAGE](https://openreview.net/pdf?id=cuvga_CiVND) |      |  |
| [R3F](https://arxiv.org/abs/2008.03156)  |      |  |
| [EWC](https://arxiv.org/abs/1612.00796)  |      |  |

### Running with your own configuration of hyper-parameters

First see `utils.py` and add your task, then run

```
python3 finetune_xlmr.py --task {your task} \
--lr {learning rate} --bs {batch size} --regularizer {ewc or r3f} --epoch {num epochs}
```

The arguments should be self explainatory.

## Supported Tricks

- EWC https://arxiv.org/abs/1612.00796
- R3F https://arxiv.org/abs/2008.03156
- SAGE https://arxiv.org/abs/2202.02664
