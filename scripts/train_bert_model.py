from __future__ import absolute_import, division, print_function

import click
import logging
import os
import random
import copy

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm, trange

from transformers import (BertConfig,BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup
# from radam import AdamW

from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel

def compute_metrics(task_name, preds, labels, every_type=False):
    label_list = ('Mechanism', 'Effect', 'Advise', 'Int.')
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
    result = {
        "Precision": p,
        "Recall": r,
        "microF": f
    }
    if every_type:
        for i, label_type in enumerate(label_list):
            p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
            result[label_type + ' Precision'] = p
            result[label_type + ' Recall'] = r
            result[label_type + ' F'] = f
    return result

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, args, config, gnn_config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(args.dropout_prob)
    
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':nn.GELU()}
        self.activation = activations[args.activation]

        if args.use_cnn:
            self.conv_list = nn.ModuleList([nn.Conv1d(config.hidden_size+2*args.pos_emb_dim, config.hidden_size, w, padding=(w-1)//2) for w in args.conv_window_size])
            self.pos_emb = nn.Embedding(2*args.max_seq_length, args.pos_emb_dim, padding_idx=0)
            
        if args.middle_layer_size == 0:
            self.classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, config.num_labels)
        else:
            self.middle_classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, args.middle_layer_size)
            self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        self.init_weights()
        
        if args.use_cnn:
            self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)

        self.bert = BertModel.from_pretrained(args.model_name_or_path)

        self.use_cnn = args.use_cnn
        self.middle_layer_size = args.middle_layer_size

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        if self.use_cnn:
            relative_dist1 *= attention_mask
            relative_dist2 *= attention_mask
            pos_embs1 = self.pos_emb(relative_dist1)
            pos_embs2 = self.pos_emb(relative_dist2)
            conv_input = torch.cat((outputs[0], pos_embs1, pos_embs2), 2)
            conv_outputs = []
            for c in self.conv_list:
                conv_output = self.activation(c(conv_input.transpose(1,2)))
                conv_output, _ = torch.max(conv_output, -1)
                conv_outputs.append(conv_output)
            pooled_output = torch.cat(conv_outputs, 1)
        
        pooled_output = self.dropout(pooled_output)
        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt




logger = logging.getLogger(__name__)

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, train_dataset, model, tokenizer, desc_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
#     train_dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.parameter_averaging:
        storage_model = copy.deepcopy(model)
        storage_model.zero_init_params()
    else:
        storage_model = None

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)
    print(model)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    bestF1 = 0.0
    #for _ in train_iterator:
    for epoch, _ in enumerate(train_iterator, start=1):
        for step, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'relative_dist1': batch[3],
                      'relative_dist2': batch[4],
                      'labels':         batch[5],}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if not args.parameter_averaging:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, desc_tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if (args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    print("Saving model checkpoint to %s", output_dir)

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            if args.parameter_averaging:
                storage_model.accumulate_params(model)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_during_training:
            prefix = 'epoch' + str(epoch)
            output_dir = os.path.join(args.output_dir, prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.parameter_averaging:
                storage_model.average_params()
                result = evaluate(args, storage_model, tokenizer, desc_tokenizer, prefix=prefix)
                storage_model.restore_params()
            else:
                results = evaluate(args, model, tokenizer, desc_tokenizer, prefix=prefix)
            if results['microF'] > bestF1:
                print("New high record!")
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-epoch{}'.format(epoch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                print("Saving model checkpoint to %s", output_dir)
                bestF1 = results['microF']
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    #return global_step, tr_loss / global_step
    return global_step, tr_loss / global_step, storage_model


#def evaluate(args, model, tokenizer, prefix=""):
def evaluate(args, model, tokenizer, desc_tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if args.feature_test_pt_path:
            eval_dataset = torch.load(args.feature_test_pt_path, weights_only=False)
        else:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, desc_tokenizer, evaluate=True, data_type="test")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        print("***** Running evaluation {} *****".format(prefix))
        print("  Num examples = %d", len(eval_dataset))
        print("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                          'labels':         batch[5],}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        try:
            np.save(os.path.join(args.output_dir, 'preds'), preds)
            np.save(os.path.join(args.output_dir, 'labels'), out_label_ids)
        except:
            print("Np.save is brokening... No problem")
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, tokenizer, desc_tokenizer, evaluate=False, data_type='no'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        # label_list = processor.get_labels()
        label_list = ['false', 'advise', 'effect',  'mechanism', 'int']
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        if data_type=="train":
            examples = torch.load("examples_train.pt")
            # Check to change label if needed
            lb_list = ['false', 'advise', 'effect',  'mechanism', 'int']
            for e_idx in range(len(examples)):
                if examples[e_idx].label == 'negative':
                    examples[e_idx].label = 'false'
                elif examples[e_idx].label in [0,1]:
                    examples[e_idx].label = lb_list[examples[e_idx].label]
            
        elif data_type=="test":
            examples = torch.load("examples_test.pt")
            # Check to change label if needed
            lb_list = ['false', 'advise', 'effect',  'mechanism', 'int']
            for e_idx in range(len(examples)):
                if examples[e_idx].label == 'negative':
                    examples[e_idx].label = 'false'
                elif examples[e_idx].label in [0,1]:
                    examples[e_idx].label = lb_list[examples[e_idx].label]
        else:
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Drug Description
    desc_max_seq_length = args.desc_max_seq_length
    desc_processor = processors['desc']()
    output_mode = output_modes[task]

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Get Position index
    drug_id = tokenizer.vocab['drug']
    one_id = tokenizer.vocab['##1']
    two_id = tokenizer.vocab['##2']

    all_input_ids = [f.input_ids for f in features]
    # print(len(all_input_ids))
    # print(all_input_ids[0].shape)
    all_entity1_pos= []
    all_entity2_pos= []
    for input_ids in all_input_ids:
        entity1_pos = args.max_seq_length-1 
        entity2_pos = args.max_seq_length-1 
        for i in range(args.max_seq_length):
            if input_ids[i] == drug_id and input_ids[i+1] == one_id:
                entity1_pos = i
            if input_ids[i] == drug_id and input_ids[i+1] == two_id:
                entity2_pos = i
        all_entity1_pos.append(entity1_pos)
        all_entity2_pos.append(entity2_pos)
    assert len(all_input_ids) == len(all_entity1_pos) == len(all_entity2_pos)

    range_list = list(range(args.max_seq_length, 2*args.max_seq_length))
    all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
    all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    all_desc1_ii = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc1_am = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc1_tti = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_ii = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_am = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_desc2_tti = torch.tensor([f.input_ids for f in features], dtype=torch.long)

    # Fingerprint
    fingerprint_indices = torch.tensor(list(range(len(features))), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_relative_dist1, all_relative_dist2,
                            all_desc1_ii, all_desc1_am, all_desc1_tti,
                            all_desc2_ii, all_desc2_am, all_desc2_tti,
                            fingerprint_indices,
                            all_labels)
    return dataset


class Args:
    def __init__(self, old_args):
        self.data_dir = old_args.data_dir
        self.model_type = old_args.model_type
        self.model_name_or_path = old_args.model_name_or_path
        self.task_name = old_args.task_name
        self.output_dir = old_args.output_dir
        self.config_name = old_args.config_name
        self.tokenizer_name = old_args.tokenizer_name
        self.cache_dir = old_args.cache_dir
        self.max_seq_length = old_args.max_seq_length
        self.do_train = old_args.do_train
        self.do_eval = old_args.do_eval
        self.evaluate_during_training = old_args.evaluate_during_training
        self.do_lower_case = old_args.do_lower_case
        self.per_gpu_train_batch_size = old_args.per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = old_args.per_gpu_eval_batch_size
        self.gradient_accumulation_steps = old_args.gradient_accumulation_steps
        self.learning_rate = old_args.learning_rate
        self.weight_decay = old_args.weight_decay
        self.adam_epsilon = old_args.adam_epsilon
        self.max_grad_norm = old_args.max_grad_norm
        self.num_train_epochs = old_args.num_train_epochs
        self.max_steps = old_args.max_steps
        self.warmup_steps = old_args.warmup_steps
        self.logging_steps = old_args.logging_steps
        self.save_steps = old_args.save_steps
        self.eval_all_checkpoints = old_args.eval_all_checkpoints
        self.no_cuda = old_args.no_cuda
        self.overwrite_output_dir = old_args.overwrite_output_dir
        self.overwrite_cache = old_args.overwrite_cache
        self.seed = old_args.seed
        self.tpu = old_args.tpu
        self.tpu_ip_address = old_args.tpu_ip_address
        self.tpu_name = old_args.tpu_name
        self.xrt_tpu_config = old_args.xrt_tpu_config
        self.fp16 = old_args.fp16
        self.fp16_opt_level = old_args.fp16_opt_level
        self.local_rank = old_args.local_rank
        self.server_ip = old_args.server_ip
        self.server_port = old_args.server_port
        self.parameter_averaging = old_args.parameter_averaging
        self.dropout_prob = old_args.dropout_prob
        self.middle_layer_size = old_args.middle_layer_size
        self.use_cnn = old_args.use_cnn
        self.conv_window_size = old_args.conv_window_size
        self.pos_emb_dim = old_args.pos_emb_dim
        self.activation = old_args.activation
        self.use_desc = old_args.use_desc
        self.desc_max_seq_length = old_args.desc_max_seq_length
        self.desc_conv_window_size = old_args.desc_conv_window_size
        self.desc_conv_output_size = old_args.desc_conv_output_size
        self.desc_layer_hidden = old_args.desc_layer_hidden
        self.use_mol = old_args.use_mol
        self.fingerprint_dir = old_args.fingerprint_dir
        self.molecular_vector_size = old_args.molecular_vector_size
        self.gnn_layer_hidden = old_args.gnn_layer_hidden
        self.gnn_layer_output = old_args.gnn_layer_output
        self.gnn_mode = old_args.gnn_mode
        self.gnn_activation = old_args.gnn_activation
        self.pretrained_dir = old_args.pretrained_dir
        self.pretrained_gnn_dir = old_args.pretrained_gnn_dir
        self.pretrained_desc_dir = old_args.pretrained_desc_dir
        self.freeze_pretrained_parameters = old_args.freeze_pretrained_parameters
        self.feature_train_pt_path = old_args.feature_train_pt_path
        self.feature_test_pt_path = old_args.feature_test_pt_path
        self.device = old_args.device


@click.command()
@click.option("--yaml_path", required=True, type=str, help="Path to the yaml config")
def main(yaml_path):
    import yaml
    from ddi_kt_2024.utils import DictAccessor
    with open(yaml_path, 'r') as f:
        args = DictAccessor(yaml.safe_load(f))
    args = Args(args)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.output_mode = 'classification'
    num_labels= 5

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    desc_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class(args, config, None)

    if not args.do_train:
        global_step = 0
        model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'state_dict')))
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    print("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.feature_train_pt_path:
            train_dataset = torch.load(args.feature_train_pt_path, weights_only=False)
        else:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, desc_tokenizer, evaluate=False, data_type="train")
        global_step, tr_loss, storage_model =  train(args, train_dataset, model, tokenizer, desc_tokenizer)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        print("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'state_dict'))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.parameter_averaging:
            #storage_model.load_state_dict(torch.load(os.path.join('/mnt/output/foo', 'state_dict_epoch5')))
            storage_model.average_params()
            result = evaluate(args, storage_model, tokenizer, desc_tokenizer, prefix="")
        else:
            #result = evaluate(args, model, tokenizer, prefix="")
            result = evaluate(args, model, tokenizer, desc_tokenizer, prefix="")
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
    return results

if __name__ == "__main__":
    main()