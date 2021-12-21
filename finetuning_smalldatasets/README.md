# Finetuning strategies

This repo contains code for finetuning strategies for small datasets. 
``` data ``` directory contains the data for the various tasks uner the respective task folder. 

To download CoLA,
```
cd CoLA/data
wget https://dl.fbaipublicfiles.com/glue/data/CoLA.zip
unzip CoLA.zip
```

### Train

command to run train.py with default strategy (linear_scheduler)

```
cd CoLA
python train.py --output model_test --modeltype bert --overwrite true --do_train True
```

To train with other strategy pass the strategy value (for example - layerwise_lrd)

```
cd CoLA
python train.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
```

## To train with layer initialization

command to run train_layerinit.py with default strategy (linear_scheduler)
```
cd CoLA
python train_layerinit.py --output model_test --modeltype bert --overwrite true --do_train True
```

To train with other strategy pass the strategy value (for example - layerwise_lrd)
```
python train_layerinit.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
```
## To train with R-Drop 

command to run train_rdrop.py with default strategy (linear_scheduler)
```
cd CoLA
python train_rdrop.py --output model_test --modeltype bert --overwrite true --do_train True
```

To train with other strategy pass the strategy value (for example - layerwise_lrd)
```
python train_rdrop.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
```




## To train with SWA and layer initialization 

command to run train_swa.py with default strategy (linear_scheduler)
```
cd CoLA
python train_swa.py --output model_test --modeltype bert --overwrite true --do_train True
```

To train with other strategy pass the strategy value (for example - layerwise_lrd)
```
python train_swa.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
```
### visualization
Tensorboard can be used with pytorch to visualise the train and valid loss. 
By default, default `log_dir` is "runs" in tensorboard.  We will be specifying a directory "CoLA_experiments' as the 'log_dir'. ``` writer = SummaryWriter('runs/CoLA_experiments')```

To view the train and valid loss, 
```
tensorboard --logdir=runs/CoLA_experiments
```
### Evaluate 
command to evaluate and write the results in the output folder.
```
cd CoLA
python train.py --output model_nov17 --modeltype bert --overwrite true --do_eval True
```
### Layer-wise Learning Rate Decay (LLRD)
This is
accomplished by setting the learning rate of the top layer and using a multiplicative decay rate to
decrease the learning rate layer-by-layer from top to bottom.

A learning rate of 3.5e-6 for the top layer is used and a multiplicative decay rate of 0.9 is applied to the learning rate to decrease the learning rate layer-by-layer from top to bottom. It will result in the bottom layers (embeddings and layer0) having a learning rate roughly close to 1e-6. Pooler or regressor layer has 3.6e-6 as learning rate, a learning rate that is slightly higher than the top layer.