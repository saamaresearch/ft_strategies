#command to run
#python train_llrd.py -output model_nov17 -modeltype bert -overwrite true

# max_length is set as 256, because df_train.sentences.map(lambda x: len(x)).max() # yielded 231.
# https://pytorch.org/docs/stable/tensorboard.html for tensorbard usage

import os
import json
import torch
import random
import logging
import argparse
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import cuda

from sklearn.metrics import matthews_corrcoef

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader,SequentialSampler,RandomSampler

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AdamW,  get_linear_schedule_with_warmup

from strategy_finetuning import linear_scheduler, layerwise_lrd, grouped_layerwise_lrd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--output", default=None, type=str, required=True, help="Output folder where model weights, metrics, preds will be saved")
parser.add_argument("--overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")

parser.add_argument("--modeltype", default=None, type=str, help="model used [bert ]", required=True)
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

parser.add_argument("--train_batch_size", default=8, type=int, help="Train batch size for training.")
parser.add_argument("--valid_batch_size", default=8, type=int, help="Valid batch size for training.")
#parser.add_argument("-test_batch_size", default=8, type=int, help="Test batch size for evaluation.")
parser.add_argument("--strategy", default="linear_scheduler" , type=str, help="strategy for finetuning ['linear_scheduler','layerwise_lrd','grouped_layerwise_lrd']")
parser.add_argument("--lr", default=3.5e-6 , type=float, help="learning rate for llrd")
parser.add_argument("--head_lr", default=3.6e-6 , type=float, help="learning rate for head")
parser.add_argument("--reinit_n_layers", default=3, type=int, help="Number of layers (close to the output) to be initialized ")
parser.add_argument("--swa_lr", default=2e-6 , type=float, help="learning rate for SWA")
parser.add_argument("--swa_start", default=3 , type=float, help="to SWA average of the parameters at this epoch number")

parser.add_argument("--epochs", default=1, type=int, help="epochs for training")
parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--do_train', type=str, default=False, help="set it to True to train")
parser.add_argument('--do_eval', type=str, default=False, help="set it to True to train")
parser.add_argument('--seed', type=int, default=42, help="set the seed for reproducibility")

args = parser.parse_args()

if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
    raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output))
if not os.path.exists(args.output):
    os.makedirs(args.output)



if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



MODEL_CLASSES = {
    'bert': "bert-base-uncased"
}

set_seed(args.seed)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/CoLA_experiments')

cols = ["0","label","1","sentence"]
df_train = pd.read_csv("data/CoLA/train.tsv", sep='\t', names=cols)
df_train.drop(["0","1"], axis=1, inplace=True)

df_valid = pd.read_csv("data/CoLA/dev.tsv", sep='\t', names=cols)
df_valid.drop(["0","1"], axis=1, inplace=True)

# # labels are all -1 in test.tsv ( Since test set doesnt have labels, we will be using dev as test)
# df_test = pd.read_csv("data/CoLA/dev.tsv", sep='\t')
# df_test.drop(["index"], axis=1, inplace=True)
# df_test = df_test[0:50]

config = AutoConfig.from_pretrained(MODEL_CLASSES[args.modeltype])
tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASSES[args.modeltype], do_lower_case=args.do_lower_case)
model = AutoModel.from_pretrained(MODEL_CLASSES[args.modeltype], config=config)

n = config.hidden_size
num_labels = 2 

# # Creating the customized model, by adding a drop out and a dense layer on top of bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
            
    def __init__(self, model, reinit_n_layers=0):        
        super().__init__() 
        self.l1 = model   
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(n, num_labels)
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()            
            
    def _do_reinit(self):
        # Re-init pooler.
        self.l1.pooler.dense.weight.data.normal_(mean=0.0, std=self.l1.config.initializer_range)
        self.l1.pooler.dense.bias.data.zero_()
        for param in self.l1.pooler.parameters():
            param.requires_grad = True

        logging.info(f"Re-initializing {self.reinit_n_layers} layers from the output")
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):            
            self.l1.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
            
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.l1.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)        

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = BERTClass(model, args.reinit_n_layers)
model.to(device)



if args.strategy == 'linear_scheduler':
    logging.info("Fine Tuning with linear_scheduler which is the default for strategy")
    train_examples = df_train.shape[0]
    num_train_optimization_steps = int(train_examples /args.train_batch_size / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer_grouped_parameters = linear_scheduler(model, args.weight_decay)
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps)
elif args.strategy == 'layerwise_lrd':
    logging.info(f"Fine Tuning with layerwise_llrd strategy with default head_lr of {args.head_lr}")
    optimizer_grouped_parameters = layerwise_lrd(model, args.weight_decay, args.lr, args.head_lr)
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr,eps=args.adam_epsilon)
elif args.strategy == 'grouped_layerwise_lrd':
    logging.info("Fine Tuning with grouped_layerwise_lrd strategy")
    optimizer_grouped_parameters = grouped_layerwise_lrd(model, args.weight_decay, args.lr)
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr,eps=args.adam_epsilon)

swa_model = AveragedModel(model).to(device)
swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

class CustomDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe
        self.sentence = dataframe.sentence
        if 'label' in dataframe.columns:
            self.targets = dataframe.label
        else:
            self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):     
        return  self.sentence[index], self.targets[index]

def process_batch(dataset,tokenizer,max_len=args.max_seq_length):

    sentences = [sentence for sentence,target in dataset]
    targets = [target for sentence,target in dataset]
    sent_tokens = tokenizer(sentences,truncation=True,padding="max_length",max_length=args.max_seq_length)

    input_ids = sent_tokens['input_ids']
    attention_mask = sent_tokens['attention_mask'] 
    token_type_ids = sent_tokens['token_type_ids']
    return torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(token_type_ids), torch.tensor(targets)
    

def train_dataloader(train_dataset):
    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(
        process_batch,
        tokenizer=tokenizer,
        max_len=args.max_seq_length
        )
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size,
                                sampler=train_sampler,
                                collate_fn=model_collate_fn)
    return train_dataloader

def valid_dataloader(valid_dataset):
    valid_sampler = SequentialSampler(valid_dataset)
    model_collate_fn = functools.partial(
        process_batch,
        tokenizer=tokenizer,
        max_len=args.max_seq_length
        )
    valid_dataloader = DataLoader(valid_dataset,
                                batch_size=args.valid_batch_size,
                                sampler=valid_sampler,
                                collate_fn=model_collate_fn)
    return valid_dataloader

# def test_dataloader(test_dataset):
#   test_sampler = SequentialSampler(test_dataset)
#   model_collate_fn = functools.partial(
#     process_batch,
#     tokenizer=tokenizer,
#     max_len=args.max_seq_length
#     )
#   test_dataloader = DataLoader(test_dataset,
#                               batch_size=args.test_batch_size,
#                               sampler=test_sampler,
#                               collate_fn=model_collate_fn)
#   return test_dataloader

train_dataset = CustomDataset(df_train)
train_dataloader = train_dataloader(train_dataset)

valid_dataset = CustomDataset(df_valid)
valid_dataloader = valid_dataloader(valid_dataset)

# test_dataset = CustomDataset(df_test)
# test_dataloader = test_dataloader(test_dataset)

if args.do_train:
    # train_loss = [] # Uncomment this only for debugging 
    def train(epoch):
        logger.info(f"Training on train dataset for epoch number - {epoch}")
        model.train()
        running_loss = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            *inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if epoch > args.swa_start:            
                    swa_model.update_parameters(model)  # To update parameters of the averaged model.
                    swa_scheduler.step() # Switch to SWALR.
                elif args.strategy == 'linear_scheduler':
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            # logger.info(f"Step Loss is {loss.item()}")

        avg_loss = sum(running_loss)/len(running_loss)
        logger.info(f"Average epoch train Loss for epoch number {epoch} is {avg_loss}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
    #   train_loss.append(avg_loss)
    # logger.info(f"Train_loss is {train_loss}")
    writer.flush()

    # valid_loss = [] # uncomment this only for debugging
    def valid(epoch):
        logger.info(f"Validation on valid dataset for epoch number - {epoch}")
        model.eval()
        running_loss = []
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            *inputs, targets = batch
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)
            running_loss.append(loss.item())
            # logger.info(f"Step Loss is {loss.item()}")

        avg_loss = sum(running_loss)/len(running_loss)
        logger.info(f"Average epoch valid Loss for epoch number {epoch} is {avg_loss}")
        writer.add_scalar("Loss/valid", avg_loss, epoch)
    #   valid_loss.append(avg_loss)
    # logger.info(f"Valid_loss is {valid_loss}")
    writer.flush()

    for epoch in range(args.epochs):
        train(epoch)
        valid(epoch)

    torch.optim.swa_utils.update_bn(train_dataloader, swa_model)

    output_model =f"{args.output}/model_cola.pth"

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_model)
    tokenizer.save_pretrained(args.output)
    config.save_pretrained(args.output)

### Evaluation 
if args.do_eval:
    output_model =f"{args.output}/model_cola.pth"
    checkpoint = torch.load(output_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def predict():
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_dataloader, 0)):
                batch = tuple(t.to(device) for t in batch)
                *inputs, targets = batch
                outputs = model(*inputs)
                outputs_softmax = torch.log_softmax(outputs, dim = 1)
                outputs_cls = torch.argmax(outputs_softmax, dim = 1)   
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs_cls.cpu().detach().numpy().tolist())
        
        return  fin_targets, fin_outputs

    y_true, y_pred = predict()
    score = matthews_corrcoef(y_true, y_pred)
    with open(f"{args.output}/predictions.json", "w") as outfile:
        json.dump(y_pred, outfile)
    with open(f"{args.output}/results.txt", "w") as file:
        file.write(f"The mathew corr coef is {score}")
