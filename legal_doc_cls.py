# -*- coding: utf-8 -*-

""" 
Created at 2019-08-09 15:47:19
==============================
legal_doc_cls 
"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, HIBERT
from pytorch_pretrained_bert.optimization import BertAdam
import random
import numpy as np
import json
import math
from torch.utils.data import Dataset, TensorDataset, DataLoader
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class CriminalDataSet(Dataset):
    def __init__(self, input_ids, attention_mask, sentence_att_mask, key_sentence_labels, penalty, prison):
        self.input_ids, self.attention_mask, self.sentence_att_mask, self.key_sentence_labels, self.penalty, self.prison = input_ids, attention_mask, sentence_att_mask, key_sentence_labels, penalty, prison

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.sentence_att_mask[index], \
               self.key_sentence_labels[index], self.penalty[index], self.prison[index]


def get_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        examples = []
        for line in f.readlines():
            sub_data = json.loads(line)
            sentence = []
            annotation = []
            for sent in sub_data['content']:
                sentence.append(sent['sent'])
                annotation.append(sent['label'])
            penalty = sub_data['penalty']
            prison = sub_data['prison']
            examples.append({'context': sentence, 'annotation': annotation, 'penalty': penalty, 'prison': prison})
        return examples


def handlePenalty(penalty):
    penalty = max(0, penalty)
    maxPenalty = math.ceil(penalty / 1000)
    if maxPenalty > 5 and maxPenalty <= 10:
        maxPenalty = 6
    elif maxPenalty > 10 and maxPenalty <= 500:
        maxPenalty = 7
    elif maxPenalty > 500:
        maxPenalty = 8
    return maxPenalty


def handlePrison(prison):
    prison = max(0, prison)
    if prison == 0:
        prison = 0
    elif prison > 0 and prison <= 30:
        prison = 1
    elif prison > 30 and prison <= 90:
        prison = 2
    elif prison > 90 and prison <= 180:
        prison = 3
    elif prison > 180 and prison <= 365:
        prison = 4
    elif prison > 365 and prison <= 1095:
        prison = 5
    elif prison > 1095 and prison <= 1825:
        prison = 6
    elif prison > 1825 and prison <= 5475:
        prison = 7
    elif prison > 5475 and prison <= 9125:
        prison = 8
    return prison


def convert_examples_to_features(examples, max_seq_length, max_sent_num, tokenizer):
    input_ids = []
    input_mask = []
    sent_masks = []
    penalties = []
    prisons = []
    annotations = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        masks = []
        for sent in example['context']:
            sub_token = tokenizer.tokenize(sent)
            if len(sub_token) > max_seq_length - 2:
                sub_token = sub_token[0:max_seq_length - 2]
            sub_tokens = []
            sub_tokens.append("[CLS]")
            for token in sub_token:
                sub_tokens.append(token)
            sub_tokens.append("[SEP]")
            mask = [1] * len(sub_tokens)
            while len(sub_tokens) < max_seq_length:
                sub_tokens.append("[PAD]")
                mask.append(0)
            masks.append(mask)
            tokens.append(tokenizer.convert_tokens_to_ids(sub_tokens))
        sentence_annotation = example['annotation']
        if len(tokens) > max_sent_num:
            tokens = tokens[:max_sent_num]
            sentence_annotation = sentence_annotation[:max_sent_num]
            masks = masks[:max_sent_num]
        sent_mask = [1] * len(tokens)
        while len(tokens) < max_sent_num:
            tokens.append([0] * max_seq_length)
            sentence_annotation.append(0)
            sent_mask.append(0)
            masks.append([0] * max_seq_length)

        input_ids.append(tokens)
        input_mask.append(masks)
        sent_masks.append(sent_mask)
        penalties.append(handlePenalty(example['penalty']))
        prisons.append(handlePrison(example['prison']))
        annotations.append(sentence_annotation)

    return input_ids, input_mask, sent_masks, penalties, prisons, annotations


def val(model, processor, args, label_list, max_seq_length, max_sent_num, tokenizer, device):
    val_path = ''
    val_examples = get_data(val_path)
    input_ids, input_mask, sent_masks, penalties, prisons, annotations = convert_examples_to_features(val_examples,
                                                                                                      max_seq_length,
                                                                                                      max_sent_num,
                                                                                                      tokenizer)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(input_mask, dtype=torch.long)
    sentence_att_mask = torch.tensor(sent_masks, dtype=torch.long)
    penalty = torch.tensor(penalties, dtype=torch.long)
    prison = torch.tensor(prisons, dtype=torch.long)
    key_sentence_labels = torch.tensor(annotations, dtype=torch.long)

    val_data = TensorDataset(input_ids, attention_mask, sentence_att_mask, key_sentence_labels, penalty, prison)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    model.eval()

    with torch.no_grad():
        for input_ids, input_mask, sent_masks, penalty, prison in val_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            sent_masks = sent_masks.to(device)
            label_ids = label_ids.to(device)

            logits = model(input_ids, input_mask, sent_masks)

            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    print(len(gt))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print(f1)

    return f1


def main():
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epoch = 32
    learning_rate = 5e-5
    warmup_proportion = 0.1

    max_seq_length = 20
    max_sent_num = 3

    train_path = '标注数据.txt'
    train_examples = get_data(train_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    input_ids, input_mask, sent_masks, penalties, prisons, annotations = convert_examples_to_features(train_examples,
                                                                                                      max_seq_length,
                                                                                                      max_sent_num,
                                                                                                      tokenizer)
    # print(np.array(input_ids).shape, np.array(input_mask).shape, np.array(sent_masks).shape, np.array(penalties).shape,
    #       np.array(annotations).shape)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(input_mask, dtype=torch.long)
    sentence_att_mask = torch.tensor(sent_masks, dtype=torch.long)
    penalty = torch.tensor(penalties, dtype=torch.long)
    prison = torch.tensor(prisons, dtype=torch.long)
    key_sentence_labels = torch.tensor(annotations, dtype=torch.long)
    # print(key_sentence_labels.size())
    train_data = TensorDataset(input_ids, attention_mask, sentence_att_mask, key_sentence_labels, penalty, prison)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # model = BertModel.from_pretrained('bert-base-chinese')
    model = HIBERT.from_pretrained('bert-base-chinese', doc_label=9, sentence_key_label=2)
    model.to(device)
    model.train()

    # Prepare optimizer
    # if args.optimize_on_cpu:
    if not torch.cuda.is_available():
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    t_total = int(
        len(train_examples) / batch_size * max_epoch)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=t_total)

    best_score = 0
    flags = 0

    for epoch in range(max_epoch):
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, sent_masks, annotations, penalty, prison = batch
            print(input_ids.size(), input_mask.size(), sent_masks.size(), annotations.size(), penalty.size(),
                  prison.size())
            loss = model(input_ids, input_mask, sent_masks, annotations, penalty)  # penalty cls
            # loss = model(input_ids, input_mask,sent_masks,annotations, prison) # prison cls
            loss.backward()
            optimizer.step()
            model.zero_grad()
            print(loss)
    # f1 = val(model, processor, args, label_list, tokenizer, device)
    # if f1 > best_score:
    #     best_score = f1
    #     print('*f1 score = {}'.format(f1))
    #     flags = 0
    #     checkpoint = {
    #         'state_dict': model.state_dict()
    #     }
    #     torch.save(checkpoint, args.model_save_pth)
    # else:
    #     print('f1 score = {}'.format(f1))
    #     flags += 1
    #     if flags >= 6:
    #         break

    # with torch.no_grad():
    #     for i, (inputs, annotation, label_result, segments_ids) in enumerate(trainDataLoader):
    #         loss = model(inputs, annotation, label_result, segments_ids)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()


if __name__ == '__main__':
    main()
    # # Convert inputs to PyTorch tensors
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    #
    # # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-chinese')
    # model.eval()
    #
    # # If you have a GPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')
    #
    # # Predict hidden states features for each layer
    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # # We have a hidden states for each of the 12 layers in model bert-base-uncased
    # assert len(encoded_layers) == 12
    #
    # # Load pre-trained model (weights)
    # model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    # model.eval()
    #
    # # If you have a GPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')
