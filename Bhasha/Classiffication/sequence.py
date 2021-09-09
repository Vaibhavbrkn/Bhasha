from typing import List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
import os
from ..utils.metrics import Metrics
from transformers import (
    BertTokenizer, BertForSequenceClassification, AutoTokenizer,
    AlbertForSequenceClassification, AlbertTokenizer,
    BartForSequenceClassification, BartTokenizer,
    BigBirdForSequenceClassification, BigBirdTokenizer,
    BigBirdPegasusForSequenceClassification, PegasusTokenizer,
    CamembertForSequenceClassification, CamembertTokenizer,
    CanineForSequenceClassification, CanineTokenizer,
    ConvBertForSequenceClassification, ConvBertTokenizer,
    DebertaForSequenceClassification, DebertaTokenizer,
    DebertaV2ForSequenceClassification, DebertaV2Tokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    ElectraForSequenceClassification, ElectraTokenizer,
    FlaubertForSequenceClassification, FlaubertTokenizer,
    FunnelForSequenceClassification, FunnelTokenizer,
    GPT2ForSequenceClassification, GPT2Tokenizer,
    GPTNeoForSequenceClassification,
    IBertForSequenceClassification,
    LEDForSequenceClassification, LEDTokenizer,
    LayoutLMForSequenceClassification, LayoutLMTokenizer,
    LayoutLMv2ForSequenceClassification, LayoutLMv2Tokenizer,
    LongformerForSequenceClassification, LongformerTokenizer,
    MBartForSequenceClassification, MBartTokenizer,
    MPNetForSequenceClassification, MPNetTokenizer,
    MegatronBertForSequenceClassification,
    MobileBertForSequenceClassification, MobileBertTokenizer,
    OpenAIGPTForSequenceClassification, OpenAIGPTTokenizer,
    ReformerForSequenceClassification, ReformerTokenizer,
    RemBertForSequenceClassification, RemBertTokenizer,
    RoFormerForSequenceClassification, RoFormerTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    SqueezeBertForSequenceClassification, SqueezeBertTokenizer,
    TapasForSequenceClassification, TapasTokenizer,
    TransfoXLForSequenceClassification, TransfoXLTokenizer,
    XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    XLNetForSequenceClassification, XLNetTokenizer,

)
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pdb
import warnings
from ..utils.LR import Scheduler, Optim

warnings.filterwarnings("ignore")


models = {
    "Bert": BertForSequenceClassification,
    "Albert": AlbertForSequenceClassification,
    "Bart": BartForSequenceClassification,
    "BigBird": BigBirdForSequenceClassification,
    "BigBirdPegasus": BigBirdPegasusForSequenceClassification,
    "Camembert": CamembertForSequenceClassification,
    "Canine": CanineForSequenceClassification,
    "ConvBert": ConvBertForSequenceClassification,
    "Deberta": DebertaForSequenceClassification,
    "DebertaV2": DebertaV2ForSequenceClassification,
    "DistilBert": DistilBertForSequenceClassification,
    "Electra": ElectraForSequenceClassification,
    "Flaubert": FlaubertForSequenceClassification,
    "Funnel": FunnelForSequenceClassification,
    "GPT2": GPT2ForSequenceClassification,
    "GPTNeo": GPTNeoForSequenceClassification,
    "IBert": IBertForSequenceClassification,
    "LED": LEDForSequenceClassification,
    "LayoutLM": LayoutLMForSequenceClassification,
    "LayoutLMv2": LayoutLMv2ForSequenceClassification,
    "Longformer": LongformerForSequenceClassification,
    "MBart": MBartForSequenceClassification,
    "MPNet": MPNetForSequenceClassification,
    "MegatronBert": MegatronBertForSequenceClassification,
    "MobileBert": MobileBertForSequenceClassification,
    "OpenAIGPT": OpenAIGPTForSequenceClassification,
    "Reformer": ReformerForSequenceClassification,
    "RemBert": RemBertForSequenceClassification,
    "RoFormer": RoFormerForSequenceClassification,
    "Roberta": RobertaForSequenceClassification,
    "SqueezeBert": SqueezeBertForSequenceClassification,
    "Tapas": TapasForSequenceClassification,
    "TransforXL": TransfoXLForSequenceClassification,
    "XLM": XLMForSequenceClassification,
    "XLMRoberta": XLMRobertaForSequenceClassification,
    "XLNet": XLNetForSequenceClassification



}

tokens = {
    'Bert': BertTokenizer,
    'Albert': AlbertTokenizer,
    "Bart": BartTokenizer,
    "BigBird": BigBirdTokenizer,
    "BigBirdPegasus":  PegasusTokenizer,
    "camembert": CamembertTokenizer,
    "Canine": CanineTokenizer,
    "ConvBert": ConvBertTokenizer,
    "Deberta": DebertaTokenizer,
    "DebertaV2": DebertaV2Tokenizer,
    "DistilBert": DistilBertTokenizer,
    "Electra": ElectraTokenizer,
    "Flaubert": FlaubertTokenizer,
    "Funnel": FunnelTokenizer,
    "GPT2": GPT2Tokenizer,
    "GPTNeo": GPT2Tokenizer,
    "IBert": RobertaTokenizer,
    "LED": LEDTokenizer,
    "LayoutLM": LayoutLMTokenizer,
    "LayoutLMv2": LayoutLMv2Tokenizer,
    "Longformer": LongformerTokenizer,
    "MBart": MBartTokenizer,
    "MPNet": MPNetTokenizer,
    "MegatronBert": BertTokenizer,
    "MobileBert": MobileBertTokenizer,
    "OpenAIGPT": OpenAIGPTTokenizer,
    "Reformer": ReformerTokenizer,
    "RemBert": RemBertTokenizer,
    "RoFormer": RoFormerTokenizer,
    "Roberta": RobertaTokenizer,
    "SqueezeBert": SqueezeBertTokenizer,
    "Tapas": TapasTokenizer,
    "TransforXL": TransfoXLTokenizer,
    "XLM": XLMTokenizer,
    "XLMRoberta": XLMRobertaTokenizer,
    "XLNet": XLNetTokenizer


}


@dataclass
class ClassificationParamsArgs:
    batch_size: int = 1
    val_size: float = .2
    max_len: int = 512
    learning_rate: float = 2e-5
    optimizer: str = 'Adam'
    # Learning Rate Schedule
    lr_schedule: str = "ReduceLROnPlateau"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_base: float = 0.01
    lr_max: float = 0.1
    lr_mode: str = 'min'
    lr_patience: int = 5
    lr_factor: float = 0.1
    lr_threshold: float = 1e-4
    lr_milestones: List[int] = field(default_factory=lambda: [5, 10])
    # Optimizer
    betas: Tuple = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False
    centered: bool = False
    alpha: float = 0.99
    rho: float = 0.9
    lr_decay: float = 0


@dataclass
class ClassificationArgs:
    pass


@dataclass
class BertData:

    text: List[str]
    labels: List[int]
    tokenizer: Any
    max_len: int = 128

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.text[idx], max_length=self.max_len,
                                pad_to_max_length=True, return_tensors="pt")
        labels = torch.tensor(self.labels[idx])

        return tokens, labels


class SequenceClassification:
    def __init__(self, model_name, model_type, device='cuda'):
        super().__init__()

        # assert model_name in list(models.keys()), "Enter correct model"
        # assert type(model_name) == str, 'model_name must be an string'
        # assert type(model_type) == str, 'model_type must be an string'
        self.device = device
        self.model = models[model_name].from_pretrained(model_type)
        self.model.to(device)
        self.tokenizer = tokens[model_name].from_pretrained(model_type)
        self.best_loss = float('inf')

    def __make_data(self, Params):

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.train_texts, self.train_labels, test_size=Params.val_size)

        train_data = BertData(train_texts, train_labels,
                              self.tokenizer, Params.max_len)
        val_data = BertData(val_texts, val_labels,
                            self.tokenizer, Params.max_len)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=Params.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=Params.batch_size)

    def train(self, train_texts, train_labels, epochs=3, metrics_step=10, save_path='model', Params=ClassificationParamsArgs):

        # print(type(train_labels))
        # assert type(train_labels) == list, "train_labels must be a list"
        # assert type(train_texts) == list, "train_texts must be a list"
        # assert len(train_labels) == len(
        #     train_texts), "train_texts and train_labels must be equal length"
        # assert type(epochs) == int, "epcohs must be a int"
        # assert epochs > 0, "epochs should be greater then 0"
        # assert type(metrics_step) == int, "metrics_step must be a int"
        # assert len(self.train_dataloader) * \
        #     epochs > metrics_step, "metrics_step should be less then total number of steps"
        # assert type(save_path) == str, "save_path must be an string"

        self.train_texts = train_texts
        self.train_labels = train_labels
        self.save_path = save_path

        self.epochs = epochs
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        self.__make_data(Params)
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=Params.learning_rate)
        optimizer = Optim(Params, self.model)
        scheduler = Scheduler(Params, optimizer, len(
            self.train_dataloader), self.epochs)

        met = Metrics(epochs, metrics_step, save_path)

        for epoch in range(self.epochs):
            self.model.train()

            met.init_train()  # Initialize Train Metrics
            pbar = tqdm(enumerate(self.train_dataloader), total=len(
                self.train_dataloader), desc='Training Epoch-{}'.format(epoch))

            for ind, inp in pbar:
                # global_step += 1
                texts, labels = inp
                if texts['input_ids'].shape[0] == 1:
                    outputs = self.model(texts['input_ids'].squeeze(0).to(
                        self.device), attention_mask=texts['attention_mask'].squeeze(0).to(self.device), labels=labels.to(self.device))
                    loss, logits = outputs[:2]
                else:
                    outputs = self.model(texts['input_ids'].squeeze().to(
                        self.device), attention_mask=texts['attention_mask'].squeeze().to(self.device), labels=labels.to(self.device))
                    loss, logits = outputs[:2]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = torch.argmax(logits.to('cpu'), 1)
                acc = accuracy_score(labels, preds)

                met.process_train(loss, acc, pbar)  # Process Metrics

            met.finish_train()  # Finish Saving Metrics for a epoch
            self.model.eval()

            pbar = tqdm(enumerate(self.val_dataloader), total=len(
                self.val_dataloader), desc='Validation Epoch-{}'.format(epoch))
            # global_step = 0
            met.init_valid()  # Initialize Train Metrics

            for ind, inp in pbar:
                with torch.no_grad():
                    # global_step += 1
                    texts, labels = inp
                    if texts['input_ids'].shape[0] == 1:
                        outputs = self.model(texts['input_ids'].squeeze(0).to(
                            self.device), attention_mask=texts['attention_mask'].squeeze(0).to(self.device), labels=labels.to(self.device))
                    else:
                        outputs = self.model(texts['input_ids'].squeeze().to(
                            self.device), attention_mask=texts['attention_mask'].squeeze().to(self.device), labels=labels.to(self.device))
                    loss, logits = outputs[:2]

                    preds = torch.argmax(logits.to('cpu'), 1)
                    acc = accuracy_score(labels, preds)
                    met.process_valid(loss, acc, pbar)

            self.save_model(met.finish_valid())

            if Params.lr_schedule != "ReduceLROnPlateau":
                scheduler.step()
            else:
                scheduler.step(met.valid_loss_list[-1])

        met.finish()
        # return self.model

    def predict(self, test_texts, max_len=128, labels=None):

        # assert type(test_texts) == list, "test_texts must be an list"
        # assert type(max_len) == int, "max_len must be an int"

        # if labels is not None:

        # assert type(labels) == list, "labels must be an list"
        # assert len(labels) == len(
        #     test_texts), "test_texts and labels should be of equal length"

        answer = {}
        pbar = tqdm(enumerate(test_texts), total=len(
            test_texts), desc='Predicting')
        acc = 0
        for ind, inp in pbar:
            tokens = self.tokenizer.encode_plus(inp, max_length=max_len,
                                                pad_to_max_length=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(tokens['input_ids'].to(
                    self.device), attention_mask=tokens['attention_mask'].to(self.device)).logits
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                if labels is not None:
                    acc += 1 if labels[ind] == preds[0] else 0
                answer[inp] = preds[0]

        if labels is not None:
            print("Accuracy : {}".format(acc/len(test_texts)))
        return answer

    def save_model(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.model.save_pretrained("{}/weight".format(self.save_path))
            self.tokenizer.save_pretrained("{}/weight".format(self.save_path))

    def deploy(self):
        pass

    def convert(self):
        pass

    def Eval(self):
        pass
