# ProDMP_RAL
This is the imitation learning code base of ProDMP, which combines the ProDMP with an encoder-decoder deep neural network.
The ProDMP alone can be used as an individual module. We implemented it together with the DMP and ProMP in https://github.com/ALRhub/MP_PyTorch. 


## Pre-requisites
conda or pip:
pytorch (ML code base), 
wandb (Online ML experiment logger)

pip:

MP_PyTorch (trajectory generator): https://pypi.org/project/mp-pytorch/,

cw2 (deploy experiment locally or on cluster): https://pypi.org/project/cw2/,

pyyaml (parse yaml file),

tabulate (utility package),

natsort (utility package),

python-mnist (utility package),

##Run Exp:
First init your wandb account: 'wandb init'

Replace my username of wandb account in all config files by your wandb account
'gelikit' -> 'my_wandb_username'

Run experiment through this way:

'python exp.py config.yaml -o --nocodecopy'

E.g. 'python digit_cw.py one_digit.yaml -o, --nocodecopy'