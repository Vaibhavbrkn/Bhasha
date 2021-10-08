from typing import List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
import os
import gradio as gr
import onnxruntime as onnxrt
from .utils import Monitor
import time
from ..utils.metrics import Metrics
from ..utils.args import TrainParams, ConfigParams, get_config
from transformers import (
    BertTokenizer, BertForSequenceClassification, AutoTokenizer,
    AlbertForSequenceClassification, AlbertTokenizer,
    BartForSequenceClassification, BartTokenizer,
    BigBirdForSequenceClassification, BigBirdTokenizer,
    BigBirdPegasusForSequenceClassification, PegasusTokenizer,
    CamembertForSequenceClassification, CamembertTokenizer,
    CanineForSequenceClassification, CanineTokenizer,
    ConvBertForSequenceClassification, ConvBertTokenizer,
    CTRLForSequenceClassification, CTRLTokenizer,
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
    'CTRL': CTRLForSequenceClassification,
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
    'CTRL': CTRLTokenizer,
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


@ dataclass
class ClassificationArgs:
    pass


@ dataclass
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
    def __init__(self, model_name, model_type, device='cuda', config=None, save_path='model',):
        super().__init__()

        assert model_name in list(models.keys()), "Enter correct model"
        assert type(model_name) == str, 'model_name must be an string'
        assert type(model_type) == str, 'model_type must be an string'
        assert type(save_path) == str, "save_path must be an string"
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        if config is not None:
            assert type(config) == getattr(
                ConfigParams, model_name), 'Wrong Config class'
            config = get_config(model_name, config)
            self.model = models[model_name](config).from_pretrained(model_type)
        else:
            self.model = models[model_name].from_pretrained(model_type)
        self.model.to(device)
        self.tokenizer = tokens[model_name].from_pretrained(model_type)
        self.best_loss = float('inf')
        self.save_path = save_path

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

    def train(self, train_texts, train_labels, epochs=3, metrics_step=10,  Params=TrainParams):

        assert type(train_labels) == list, "train_labels must be a list"
        assert type(train_texts) == list, "train_texts must be a list"
        assert len(train_labels) == len(
            train_texts), "train_texts and train_labels must be equal length"
        assert type(epochs) == int, "epcohs must be a int"
        assert epochs > 0, "epochs should be greater then 0"
        assert type(metrics_step) == int, "metrics_step must be a int"

        self.train_texts = train_texts
        self.train_labels = train_labels
        self.epochs = epochs

        self.__make_data(Params)

        assert len(self.train_dataloader) * \
            epochs > metrics_step, "metrics_step should be less then total number of steps"

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

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

    def predict(self, test_texts, max_len=128, labels=None, device='cpu'):

        if type(test_texts) == str:
            test_texts = [test_texts]

        assert type(test_texts) == list, "test_texts must be an list"
        assert type(max_len) == int, "max_len must be an int"

        if labels is not None:

            assert type(labels) == list, "labels must be an list"
            assert len(labels) == len(
                test_texts), "test_texts and labels should be of equal length"
        self.model.to(device)
        answer = {}
        pbar = tqdm(enumerate(test_texts), total=len(
            test_texts), desc='Predicting')
        acc = 0
        for ind, inp in pbar:
            tokens = self.tokenizer.encode_plus(inp, max_length=max_len,
                                                pad_to_max_length=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(tokens['input_ids'].to(
                    device), attention_mask=tokens['attention_mask'].to(device)).logits
                outputs = outputs.detach().cpu().numpy()
                print(outputs)
                preds = np.argmax(outputs, axis=1)
                if labels is not None:
                    acc += 1 if labels[ind] == preds[0] else 0
                answer[inp] = preds[0]

        if labels is not None:
            pass
        return answer

    def save_model(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.model.save_pretrained("{}/weight".format(self.save_path))
            self.tokenizer.save_pretrained("{}/weight".format(self.save_path))

    def deploy(self, onnx=None, quantized=None, original=True, labels=None, max_len=512, device='cpu'):
        outputs = gr.outputs.Label(num_top_classes=len(labels))
        if onnx is not None:
            def predict(inp):
                onnx_session = onnxrt.InferenceSession(onnx)
                mon = Monitor(time.time(), inp, self.model_name,
                              self.model_type, max_len, 'onnx', self.save_path)

                def to_numpy(tensor):
                    if tensor.requires_grad:
                        return tensor.detach().cpu().numpy()
                    return tensor.cpu().numpy()
                tokens = self.tokenizer.encode_plus(inp, max_length=max_len,
                                                    pad_to_max_length=True, return_tensors="pt")
                input_ids = tokens['input_ids'].squeeze(1)
                attention_mask = tokens['attention_mask'].squeeze(1)
                inp = {onnx_session.get_inputs()[0].name: to_numpy(input_ids),
                       onnx_session.get_inputs()[1].name: to_numpy(
                           attention_mask)
                       }
                onnx_output = onnx_session.run(None, inp)
                onnx_output = torch.tensor(onnx_output)
                preds = torch.nn.functional.softmax(onnx_output, dim=2)
                mon.finish(time.time(), labels[torch.argmax(preds)])
                return {labels[i]: float(preds[0][0][i]) for i in range(len(labels))}

        elif quantized is not None:
            def predict(inp):
                mon = Monitor(time.time(), inp, self.model_name,
                              self.model_type, max_len, 'quantized', self.save_path)
                model = torch.jit.load(quantized)
                tokens = self.tokenizer.encode_plus(inp, max_length=max_len,
                                                    pad_to_max_length=True, return_tensors="pt")
                outputs = model(tokens['input_ids'].to(
                    device), attention_mask=tokens['attention_mask'].to(device))['logits']
                preds = torch.nn.functional.softmax(outputs, dim=1)
                mon.finish(time.time(), labels[torch.argmax(preds)])
                return {labels[i]: float(preds[0][i]) for i in range(len(labels))}
        else:
            def predict(inp):
                mon = Monitor(time.time(), inp, self.model_name,
                              self.model_type, max_len, 'original', self.save_path)
                self.model.to(device)
                tokens = self.tokenizer.encode_plus(inp, max_length=max_len,
                                                    pad_to_max_length=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(tokens['input_ids'].to(
                        device), attention_mask=tokens['attention_mask'].to(device)).logits
                    outputs = outputs.detach().cpu()
                    preds = torch.nn.functional.softmax(outputs, dim=1)
                    mon.finish(time.time(), labels[torch.argmax(preds)])
                    return {labels[i]: float(preds[0][i]) for i in range(len(labels))}

        gr.Interface(fn=predict, inputs='textbox',
                     outputs=outputs).launch(share=True)

    def convert(self, name=['onnx']):
        if not os.path.isdir("{}/convert".format(self.save_path)):
            os.mkdir("{}/convert".format(self.save_path))
        self.model.to('cpu')
        self.model.eval()
        texts, labels = next(iter(self.train_dataloader))
        input_ids = texts['input_ids'].squeeze(1)
        attention_mask = texts['attention_mask'].squeeze(1)
        dummy_input = (input_ids, attention_mask)
        input_names = ["input_ids",  "attention_mask"]
        output_names = ["output"]

        if 'onnx' in name:
            torch.onnx.export(self.model, dummy_input, "{}/convert/model.onnx".format(self.save_path),
                              input_names=input_names, output_names=output_names, dynamic_axes={
                                  "input_ids": {0: "batch_size"},
                                  "attention_mask": {0: "batch_size"},
                                  "output": {0: "batch_size"}
            }, opset_version=11)

        if 'quantized' in name:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            traced_model = torch.jit.trace(
                quantized_model, dummy_input, strict=False)
            torch.jit.save(
                traced_model, "{}/convert/quantized.pt".format(self.save_path))

    def Eval(self):
        pass
