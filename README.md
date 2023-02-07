# Translationese

## Getting Started
To install the dependencies: `pip3 install -r requirements.txt`

To download the data: `bash scripts/download_data.sh`

## Reproducing
- PAWS-X `bash scripts/pawsx.sh`
- XNLI `bash scripts/xnli.sh`

To run an experiment with your own task and set of hyper-parameters:

First see `utils.py` and add your task, then run

`python3 finetune_xlmr.py --task {your task} --lr {learning rate} --bs {batch size} --regularizer {ewc or r3f}`

The arguments should be self explainatory.
