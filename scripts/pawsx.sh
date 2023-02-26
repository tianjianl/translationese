python3 finetune_xlmr.py --task pawsx --lr 0.000005 --epoch 5 --bs 32 --max_len 128 --dropconnect
python3 finetune_xlmr.py --task xnli --lr 0.000005 --epoch 5 --bs 32 --max_len 128 --dropconnect
#python3 finetune_xlmr.py --task pawsx --lr 0.000005 --epoch 5 --bs 32 --max_len 128 --sage

#python3 finetune_xlmr.py --task pawsx --lr 0.000005 --epoch 5 --bs 32 --max_len 128 --regularizer r3f
#python3 finetune_xlmr.py --task pawsx --lr 0.000005 --epoch 5 --bs 32 --max_len 128 --regularizer ewc
