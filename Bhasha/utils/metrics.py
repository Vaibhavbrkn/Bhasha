import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from math import ceil, log10


def save_checkpoint(save_path, model):

    if save_path == None:
        return

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def save_plot(x, y, title, xlab, ylab, save_pth, finished=False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure()

    if len(x) > 4:
        plt.plot(x, y, random.choice(colors))
        tic = int(min(x) + (max(x) / 10))
        plt.xticks(np.arange(min(x), tic*10, tic))
    elif len(x) == 1:
        plt.plot(x, y, random.choice(colors), marker='o')
        plt.xticks(np.arange(x[0], x[0]*4, x[0]))
    else:
        plt.plot(x, y, random.choice(colors))
        plt.xticks(np.arange(x[0], (x[1]-x[0])*5, (x[1] - x[0])))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(save_pth)


def save_global(save_path, epochs, train_loss, valid_loss, train_accuracy, valid_accuracy, finished=False):
    epochs = list(np.arange(1, epochs + 1))
    sns.set_style('darkgrid')
    if not os.path.isdir("{}/metrics".format(save_path)):
        os.mkdir("{}/metrics".format(save_path))
    save_pth = "{}/metrics".format(save_path)

    save_plot(epochs, train_loss, "Training Loss VS Epochs",
              "Epochs", "Training Loss", "{}/train_loss.png".format(save_pth), finished)

    save_plot(epochs, valid_loss, "Validation Loss VS Epochs",
              "Epochs", "Validation Loss", "{}/valid_loss.png".format(save_pth), finished)

    save_plot(epochs, train_accuracy, "Training Accuracy VS Epochs",
              "Epochs", "Training Accuracy", "{}/train_accuracy.png".format(save_pth), finished)

    save_plot(epochs, valid_accuracy, "Validation Accuracy VS Epochs",
              "Epochs", "Validation Accuracy", "{}/valid_accuracy.png".format(save_pth), finished)


def save_train_step(save_path, metrics_step, train_step,  train_step_loss,  train_step_accuracy, finished=False):
    train_step = list(
        np.arange(metrics_step, train_step + 1, metrics_step))
    sns.set_style('darkgrid')
    if not os.path.isdir("{}/metrics".format(save_path)):
        os.mkdir("{}/metrics".format(save_path))
    save_pth = "{}/metrics".format(save_path)

    save_plot(train_step, train_step_loss, "Training Loss VS Steps",
              "Steps", "Training Loss", "{}/train_loss_step.png".format(save_pth), finished)

    save_plot(train_step, train_step_accuracy, "Training Accuracy VS Steps",
              "Steps", "Training Accuracy", "{}/train_accuracy_step.png".format(save_pth), finished)


def save_valid_step(save_path, metrics_step,  valid_step, valid_step_loss, valid_step_accuracy, finished=False):
    valid_step = list(
        np.arange(metrics_step, valid_step + 1, metrics_step))

    sns.set_style('darkgrid')
    if not os.path.isdir("{}/metrics".format(save_path)):
        os.mkdir("{}/metrics".format(save_path))
    save_pth = "{}/metrics".format(save_path)

    save_plot(valid_step, valid_step_loss, "Validation Loss VS Steps",
              "Steps", "Validation Loss", "{}/valid_loss_step.png".format(save_pth), finished)

    save_plot(valid_step, valid_step_accuracy, "Validation Accuracy VS Steps",
              "Steps",  "Validation Accuracy",  "{}/valid_accuracy_step.png".format(save_pth), finished)


class Metrics:
    def __init__(self, epochs, metrics_step, save_path):
        super().__init__()
        self.epochs = epochs
        self.running_loss = 0.0
        self.valid_running_loss = 0.0
        self.train_step_loss = []
        self.valid_step_loss = []
        self.train_step_accuracy = []
        self.valid_step_accuracy = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.accuracy_list = []
        self.valid_accuracy_list = []
        self.running_accuracy = 0.0
        self.valid_runnung_accuracy = 0.0
        self.metrics_step = metrics_step
        self.ep = 0
        self.train_step = 0
        self.valid_step = 0
        self.save_path = save_path

    def init_train(self):
        self.ep += 1
        self.running_loss = 0.0
        self.step_loss = 0.0
        self.step_accuracy = 0.0
        self.running_accuracy = 0.0
        self.global_step = 0

    def process_train(self, loss, acc, pbar):
        self.global_step += 1
        self.train_step += 1
        self.running_loss += loss.item()
        local_loss = loss.item()
        self.step_loss += loss.item()
        self.running_accuracy += acc
        self.step_accuracy += acc

        pbar.set_postfix({"loss": self.running_loss / self.global_step,
                          'accuracy': self.running_accuracy / self.global_step})

        if self.train_step % self.metrics_step == 0:

            self.train_step_accuracy.append(
                self.step_accuracy/self.metrics_step)
            self.train_step_loss.append(self.step_loss/self.metrics_step)
            self.step_accuracy = 0.0
            self.step_loss = 0.0

            save_train_step(self.save_path, self.metrics_step, self.train_step,
                            self.train_step_loss, self.train_step_accuracy)

    def finish_train(self):
        self.train_loss_list.append(self.running_loss / self.global_step)
        self.accuracy_list.append(self.running_accuracy / self.global_step)

    def init_valid(self):
        self.valid_running_loss = 0.0
        self.valid_running_accuracy = 0.0
        self.step_loss = 0.0
        self.step_accuracy = 0.0
        self.global_step = 0

    def process_valid(self, loss, acc, pbar):
        self.global_step += 1
        self.valid_step += 1
        self.valid_running_loss += loss.item()
        self.step_loss += loss.item()
        self.valid_running_accuracy += acc
        self.step_accuracy += acc

        pbar.set_postfix({" Validation loss": self.valid_running_loss / self.global_step,
                          'Validation accuracy': self.valid_running_accuracy / self.global_step})

        if self.valid_step % self.metrics_step == 0:
            self.valid_step_accuracy.append(
                self.step_accuracy/self.metrics_step)
            self.valid_step_loss.append(self.step_loss/self.metrics_step)
            self.step_accuracy = 0.0
            self.step_loss = 0.0

            save_valid_step(self.save_path, self.metrics_step, self.valid_step,
                            self.valid_step_loss, self.valid_step_accuracy)

    def finish_valid(self):

        self.valid_loss_list.append(self.valid_running_loss / self.global_step)
        self.valid_accuracy_list.append(
            self.valid_running_accuracy / self.global_step)

        if self.ep == self.epochs:
            save_global(self.save_path, self.ep, self.train_loss_list, self.valid_loss_list,
                        self.accuracy_list, self.valid_accuracy_list, True)
        else:
            save_global(self.save_path, self.ep, self.train_loss_list,
                        self.valid_loss_list, self.accuracy_list, self.valid_accuracy_list, False)

        return self.valid_running_loss / self.global_step

    def finish(self):

        save_train_step(self.save_path, self.metrics_step, self.train_step,
                        self.train_step_loss, self.train_step_accuracy, True)

        save_valid_step(self.save_path, self.metrics_step, self.valid_step,
                        self.valid_step_loss, self.valid_step_accuracy, True)
