#!/usr/bin/env python3
"""
Implementation of the Model Zoo for continual learning
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


from copy import deepcopy
from datasets.build_dataset import fetch_dataclass
from utils.config import fetch_configs
from multihead import MultiHead


class ModelZoo():
    def __init__(self, args, data_conf, hp_conf):
        """
        Initialize model zoo hyper-parmeters and dataset

        params:
          - args:      Argparse arguments
          - data_conf: JSON of data configuration
          - hp_conf:   Hyper-parameter configuration
        """
        # Initialize wts / task details
        self.tasks_info = data_conf['tasks']
        self.num_tasks = len(self.tasks_info)
        self.args = args
        self.data_conf = data_conf
        self.hp_conf = hp_conf
        self.dataclass = fetch_dataclass(data_conf['dataset'])
        self.wts = np.array([1.0 for i in range(self.num_tasks)])
        self.learner_task_idx = []

        # Prediction of individual models
        self.tr_preds = {}
        self.te_preds = {}
        for t_id in range(self.num_tasks):
            self.te_preds[t_id] = []
            self.tr_preds[t_id] = []

    def add_learner(self, learner_conf):
        """
        Add a learner to the model-Zoo

        params:
          - learner_conf: List[List[int] indicating Subset of tasks to train on
        """
        model = MultiHead(self.args, self.hp_conf, learner_conf)
        net, trainmets = model.train(log_interval=200)

        # Store predictions of learner on train/test dataset (so that we can
        # discard the learner and save memory)
        tr_ret = self.fetch_predictions(net, learner_conf['tasks'], True)
        te_ret = self.fetch_predictions(net, learner_conf['tasks'], False)
        for idx, t_id in enumerate(self.learner_task_idx):
            self.tr_preds[t_id].append(tr_ret[idx])
            self.te_preds[t_id].append(te_ret[idx])

    def sample_tasks(self, rounds: int):
        """
        Sample tasks to be used to train the next learner. Sampling changes
        based on if the Model Zoo is for continual learning or regular
        multi-task learning.

        params:
          - rounds: Number of learners added to Zoo
        """
        # Randomize here so that every iteration has a different set of tasks
        # The random seed is fixed in 'train_model' in order to ensure that the
        # same dataset is sampled in every round.
        # TODO: Fix the random seed for reproducible
        np.random.seed(seed=None)

        # Number of tasks trained in the learner
        numsubtasks = min(self.args.tasks_per_round, rounds + 1)
        pr = self.wts[:rounds] / np.sum(self.wts[:rounds])
        if rounds != 0:
            learner_task_idx = np.random.choice(rounds,
                                                numsubtasks - 1,
                                                replace=False, p=pr)
        else:
            learner_task_idx = np.array([])

        # Manually add the newly seen task (although boosting-based version
        # automatically selects this task due to the the very large loss)
        learner_task_idx = np.append(learner_task_idx, int(rounds))
        learner_task_idx = np.array(learner_task_idx, dtype=np.int32)

        learner_task_info = []
        for idx in learner_task_idx:
            learner_task_info.append(self.tasks_info[idx])

        self.learner_task_info = learner_task_info
        self.learner_task_idx = learner_task_idx

        learner_conf = deepcopy(self.data_conf)
        learner_conf["tasks"] = deepcopy(learner_task_info)
        return learner_conf

    def update_wts(self, losses):
        """
        Update the sampling weights based on the losses

        params:
          - losses: List of training losses on various tasks
        """
        losses = (losses - np.mean(losses)) / np.mean(losses)
        losses = np.exp(losses)
        losses = np.clip(losses, 0.0001, 1000)

        self.wts = losses
        return losses

    def evaluate(self, rounds: int):
        """
        Evaluate on the train and test sets and log the results

        params:
          - rounds: Number of learners added to Zoo
        """
        tr_ret = self.evaluate_preds(self.tr_preds, True)
        te_ret = self.evaluate_preds(self.te_preds, False)
        info = {
            "round": rounds,
            "TrainLoss": tr_ret['Loss'],
            "TrainAcc": tr_ret['Accuracy'],
            "TestLoss": te_ret['Loss'],
            "TestAcc": te_ret['Accuracy'],
            "last_learner_tasks": list(self.learner_task_idx),
            "last_learner_weights": list(self.wts)
        }
        print(info)
        return tr_ret['Loss']

    def fetch_predictions(self, net, l_task_info, tr_flag=False):
        """
        Store the set of predictions for all tasks using the newly trained
        learner. Store the predictions so that they can be used later for
        ensembling.

        params:
          - net: Trained neural net
          - l_task_info: Description of subset of tasks that neural net was
                         trained on
          - tr_flag:     Determines whether to use train/test set
        """
        dataset = self.dataclass(args=self.args, tasks=l_task_info)

        test_loaders = []
        for t_id in range(len(l_task_info)):
            test_loaders.append(
                dataset.get_task_data_loader(t_id, 100, 6, train=tr_flag))

        task_outputs = []
        net.eval()

        with torch.inference_mode():
            for dataloader in test_loaders:
                outputs = []
                for dat, target in dataloader:
                    tasks, labels = target
                    tasks = tasks.long()
                    labels = labels.long()

                    if self.args.gpu:
                        dat = dat.cuda(non_blocking=True)
                        tasks = tasks.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                    out = net(dat, tasks)
                    out = nn.functional.softmax(out, dim=1)
                    out = out.cpu().detach().numpy()
                    outputs.append(out)
                outputs = np.concatenate(outputs)
                task_outputs.append(outputs)
        return task_outputs

    def evaluate_preds(self, preds, tr_flag):
        """
        Use the set of predictions from all learners to give the result of the
        final ensemble.
        """
        dataset = self.dataclass(args=self.args, tasks=self.tasks_info)
        criterion = nn.NLLLoss()
        numcls = len(self.tasks_info[0])

        test_loaders = []
        for t_id in range(self.num_tasks):
            test_loaders.append(
                dataset.get_task_data_loader(t_id, 100, 6, train=tr_flag))

        all_loss = []
        all_acc = []

        for task_id, dataloader in enumerate(test_loaders):
            count = 0
            acc = 0
            loss = 0

            # The ensemble averaging occurs below. If model has no prediction,
            # output uniform probabilities over the classes
            if len(preds[task_id]) == 0:
                numpts = len(dataloader.dataset.data)
                curpred = np.ones((numpts, numcls)) / numcls
            else:
                curpred = np.mean(preds[task_id], axis=0)

            for dat, target in dataloader:
                tasks, labels = target
                tasks = tasks.long()
                batch_size = int(labels.size()[0])

                if self.args.gpu:
                    dat = dat.cuda(non_blocking=True)
                    tasks = tasks.cuda(non_blocking=True)

                out = curpred[count:count + batch_size]
                out = torch.log(torch.Tensor(out))

                loss += (criterion(out, labels).item()) * batch_size

                labels = labels.cpu().numpy()
                out = out.cpu().detach().numpy()
                acc += np.sum(labels == (np.argmax(out, axis=1)))
                count += batch_size

            all_loss.append(loss / count)
            all_acc.append(acc / count)

        info = {"Loss": all_loss,
                "Accuracy": all_acc,
                "train": tr_flag}
        return info

    def train(self):
        """
        Train the Model Zoo
        """
        self.evaluate(0)
        for rounds in range(self.num_tasks):

            learner_conf = self.sample_tasks(rounds)
            self.add_learner(learner_conf)
            losses = self.evaluate(rounds + 1)
            self.update_wts(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int,
                        default=100,
                        help="Random Seed")

    parser.add_argument("--tasks_per_round", type=int,
                        default=5,
                        help="Number of sub-tasks (batch-size)")
    parser.add_argument("--epochs", type=int,
                        default=100,
                        help="Number of Epochs")

    parser.add_argument("--data_config", type=str,
                        default="./config/dataset/coarse_cifar100.yaml",
                        help="Multi-task config")
    parser.add_argument("--replay_frac", type=int,
                        default=1.0,
                        help="Fraction of samples used for replay")

    parser.add_argument("--hp_config", type=str,
                        default="./config/hyperparam/default.yaml",
                        help="Hyper parameter configuration")

    args = parser.parse_args()
    data_conf = fetch_configs(args.data_config)
    hp_conf = fetch_configs(args.hp_config)
    args.fp16 = args.gpu = torch.cuda.is_available()
    args.model = hp_conf["model"]
    args.dataset = data_conf["dataset"]
    args.data = fetch_dataclass(data_conf["dataset"])

    # Choose best implementation for functions
    # Does sacrifice exact reproducability from random seed
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Train Model Zoo
    zoo = ModelZoo(args, data_conf, hp_conf)
    zoo.train()


if __name__ == '__main__':
    main()
