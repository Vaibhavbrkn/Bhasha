from typing import List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
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
import nvidia_smi


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


def save_checkpoint(save_path, model):

    if save_path == None:
        return

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


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
    def __init__(self, model_name, model_type, device='cuda', load_model=False):
        super().__init__()
        if load_model:
            pass
        else:
            self.model = models[model_name].from_pretrained(model_type)
        self.model.to(device)
        self.device = device
        self.tokenizer = tokens[model_name].from_pretrained(model_type)
        self.best_loss = float('inf')

    def set_hyper(self, batch_size=4, val_size=0.2, max_len=128, learning_rate=2e-5):
        self.batch_size = batch_size
        self.val_size = val_size
        self.max_len = 128
        self.learning_rate = learning_rate

    def __make_data(self):

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.train_texts, self.train_labels, test_size=self.val_size)

        train_data = BertData(train_texts, train_labels,
                              self.tokenizer, self.max_len)
        val_data = BertData(val_texts, val_labels,
                            self.tokenizer, self.max_len)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size)

    def train(self, train_texts, train_labels, epochs=3, metrics_step=10, save_path='model'):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.save_path = save_path

        self.epochs = epochs
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        self.__make_data()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

        met = Metrics(epochs, metrics_step, save_path)

        for epoch in range(self.epochs):
            self.model.train()

            met.init_train()  # Initialize Train Metrics
            pbar = tqdm(enumerate(self.train_dataloader), total=len(
                self.train_dataloader), desc='Training Epoch-{}'.format(epoch))

            for ind, inp in pbar:
                # global_step += 1
                texts, labels = inp
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
                    outputs = self.model(texts['input_ids'].squeeze().to(
                        self.device), attention_mask=texts['attention_mask'].squeeze().to(self.device), labels=labels.to(self.device))
                    loss, logits = outputs[:2]

                    preds = torch.argmax(logits.to('cpu'), 1)
                    acc = accuracy_score(labels, preds)
                    met.process_valid(loss, acc, pbar)

            self.save_model(met.finish_valid())

        met.finish()
        # return self.model

    def predict(self, test_texts, max_len=128, labels=None):
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


# ToDo : EarlyStopping and callbacks as well as base model
