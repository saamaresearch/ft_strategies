python train.py --output model_test --modeltype bert --overwrite true --do_train True
python train.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
python train.py --output model_test --modeltype bert --overwrite true --do_train True --strategy grouped_layerwise_lrd


python train_layerinit.py --output model_test --modeltype bert --overwrite true --do_train True
python train_layerinit.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
python train_layerinit.py --output model_test --modeltype bert --overwrite true --do_train True --strategy grouped_layerwise_lrd


python train_swa.py --output model_test --modeltype bert --overwrite true --do_train True
python train_swa.py --output model_test --modeltype bert --overwrite true --do_train True --strategy layerwise_lrd
python train_swa.py --output model_test --modeltype bert --overwrite true --do_train True --strategy grouped_layerwise_lrd