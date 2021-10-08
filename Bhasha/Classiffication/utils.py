import time
import datetime
import os
import csv
from csv import writer


class Monitor:
    def __init__(self, start_time, inp, model_name, model_type, max_len, weight_type, file_path):
        self.start_time = start_time
        self.inp = inp
        self.model_name = model_name
        self.model_type = model_type
        self.max_len = max_len
        self.weight_type = weight_type
        self.current_time = datetime.datetime.now()
        self.file_path = file_path

    def save_file(self, latency, input_length, label, file_name, mode):
        with open(file_name, mode=mode) as csv_file:
            if mode == 'w':
                fieldnames = ['text', 'length', 'max_len', 'latency',
                              'label', 'model', 'weight', 'model_type', 'time']
                writer_object = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer_object.writeheader()
                writer_object.writerow({"text": self.inp, "length": input_length, "max_len": self.max_len,
                                        "latency": round(latency, 2), "label": label, "model": self.model_name, "weight": self.model_type, "model_type": self.weight_type, "time": self.current_time})
            else:
                writer_object = writer(csv_file)
                writer_object.writerow([self.inp, input_length, self.max_len, round(latency, 2), label,
                                        self.model_name, self.model_type, self.weight_type, self.current_time])

    def finish(self, finish_time,  label):
        latency = finish_time - self.start_time
        input_length = len(self.inp.split())

        if not os.path.isdir("{}/monitor".format(self.file_path)):
            os.mkdir("{}/monitor".format(self.file_path))

        file_name = "{}/monitor/monitor.csv".format(self.file_path)

        if os.path.exists(file_name):
            self.save_file(latency, input_length, label, file_name, 'a')
        else:
            self.save_file(latency, input_length, label, file_name, 'w')
