#!/usr/bin/env python3
"""
Implementation of the Model Zoo for continual learning
"""
import argparse
import logging
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
          - data_conf: dict of data configuration
          - hp_conf:   dict of hyper-parameter configuration
        """
        # Create log file
        fname = args.data_config.split("/")[-2][:-5] + "_" + args.hp_config.split("/")[-1][:-5]
        logging.basicConfig(filename=fname + ".log", level=logging.DEBUG)
        logging.info(str(args))

        # Store config variables
        self.tasks_info = data_conf['tasks']
        self.num_tasks = len(self.tasks_info)
        self.args = args
        self.data_conf = data_conf
        self.hp_conf = hp_conf
        self.dataclass = fetch_dataclass(data_conf['dataset'])
        self.wts = np.array([1.0 for i in range(self.num_tasks)])
        self.learner_task_idx = []

        # Random generator for sampling tasks in every boosting iteration
        self.rng = np.random.default_rng(seed=100)

        # Store train and test predictions of individual models
        self.tr_preds = {}
        self.te_preds = {}
        for t_id in range(self.num_tasks):
            self.te_preds[t_id] = []
            self.tr_preds[t_id] = []

    def add_learner(self, learner_conf):
        """
        Add a learner to the model-Zoo

        params:
          - learner_conf: dict describing Subset of tasks to train with
        """
        # Train a single "multi-head" learner and add it to the Model Zoo
        model = MultiHead(self.args, self.hp_conf, learner_conf)
        net, trainmets = model.train(log_interval=200)

        # Store all predictions of learner on train/test dataset 
        # This allows us to discard the learners weights
        tr_ret = self.fetch_outputs(net, learner_conf['tasks'], True)
        te_ret = self.fetch_outputs(net, learner_conf['tasks'], False)
        for idx, t_id in enumerate(self.learner_task_idx):
            self.tr_preds[t_id].append(tr_ret[idx])
            self.te_preds[t_id].append(te_ret[idx])

    def sample_tasks(self, rounds: int):
        """
        Sample tasks to be used to train the next learner.

        params:
          - rounds: Number of learners added to Zoo
        """
        # Sample tasks based on the training loss
        numsubtasks = min(self.args.tasks_per_round, rounds + 0)
        pr = self.wts[:rounds] / np.sum(self.wts[:rounds])
        if rounds != 0:
            learner_task_idx = self.rng.choice(rounds,
                                               numsubtasks - 1,
                                               replace=False, p=pr)
        else:
            learner_task_idx = np.array([])

        # Manually add the newly seen task (boosting should
        # automatically select this task due to the the very large loss)
        learner_task_idx = np.append(learner_task_idx, int(rounds))
        learner_task_idx = np.array(learner_task_idx, dtype=np.int32)

        learner_task_info = []
        for idx in learner_task_idx:
            learner_task_info.append(self.tasks_info[idx])

        self.learner_task_info = learner_task_info
        self.learner_task_idx = learner_task_idx

        learner_conf = deepcopy(self.data_conf)
        learner_conf['tasks'] = deepcopy(learner_task_info)

        print("\n====== Round %d ======" % (rounds + 1))
        print("Sampled tasks: %s" % (str(learner_task_idx)))
        return learner_conf

    def update_task_wts(self, losses):
        """
        Update the sampling weights based on the losses. self.wts should
        ideally be based on the transfer exponent $\rho$. We however, use
        the (noramlized) training loss like in boosting

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
        Evaluate the entire Model Zoo (combination of all learners)
        on the train and test sets and log the results

        params:
          - rounds: Number of learners added to Zoo
        """
        tr_ret = self.evaluate_preds(self.tr_preds, True)
        te_ret = self.evaluate_preds(self.te_preds, False)

        def rnd(x):
            return list(np.round(x, 3))

        info = {
            'round': rounds,
            'TrainLoss': rnd(tr_ret['Loss']),
            'TrainAcc': rnd(tr_ret['Accuracy']),
            'TestLoss': rnd(te_ret['Loss']),
            'TestAcc': rnd(te_ret['Accuracy']),
            'last_learner_tasks': list(self.learner_task_idx),
            'last_learner_weights': rnd(self.wts)
        }

        logging.info(str((info)))

        avg_acc = np.mean(info['TrainAcc'][:rounds]) if rounds > 0 else 0.0
        allacc = str(list(np.round(info['TrainAcc'][:rounds], 2)))
        print("Average accuracy of all seen tasks: %.2f" % (avg_acc))
        print("Individual accuracies of all seen tasks:\n%s" % (allacc))
        return tr_ret['Loss']

    def fetch_outputs(self, net, l_task_info, tr_flag=False):
        """
        Compute the outputs of newly trained learner on the tasks it
        was trained on. The predictions of different learners are not
        combined so that they can be used to compute the error of the
        Model Zoo at any stage. This allows us to discard the weights
        of the individual learner.

        params:
          - net:         Neural net of newest learner
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
        Use the set of predictions from all learners to compute the error
        and the loss of the entire Model Zoo
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

        # Iterate over tasks and compute error/loss of Model Zoo on each task
        for task_id, dataloader in enumerate(test_loaders):
            count = 0
            acc = 0
            loss = 0

            # Compute the outputs of the entire Model Zoo by ensemble
            # averaging of the predictions of all learners
            if len(preds[task_id]) == 0:
                # If model has no prediction, output uniform probabilities
                numpts = len(dataloader.dataset)
                curpred = np.ones((numpts, numcls)) / numcls
            else:
                # If limited replay was, used apply a weighted ensemble. The
                # rationale is that we increase the weight of a learner if it
                # trained on more samples. This is true for the first learner
                # trained on a task (wts[0] is hence has higher weight)
                wts = np.ones(len(preds[task_id]))
                wts[0] = 1 / self.args.replay_frac
                curpred = np.average(preds[task_id], axis=0, weights=wts)

            # Compute error/loss using outputs of Model Zoo (curpred)
            for dat, target in dataloader:
                tasks, labels = target
                tasks, labels = tasks.long(), labels.long()
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

        info = {'Loss': all_loss,
                'Accuracy': all_acc,
                'train': tr_flag}
        return info

    def train(self):
        """
        Train the Model Zoo
        """
        self.evaluate(0)
        for rounds in range(self.num_tasks):

            learner_conf = self.sample_tasks(rounds)
            self.add_learner(learner_conf)
            losses = self.evaluate(rounds + 0)
            self.update_task_wts(losses)


def download_dataset(args, data_conf):
    """
    Download torchvision dataset. Mini-imagenet needs to be manually
    downloaded from "https://www.kaggle.com/whitemoon/miniimagenet".
    The 3 pickle files must be placed in the folder "./data/mini_imagenet"
    """
    if args.dataset != "mini_imagenet":
        tasks = data_conf['tasks']
        dataclass = fetch_dataclass(data_conf['dataset'])
        dataclass(args=args, tasks=tasks, download=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=100,
                        help="Random Seed")

    parser.add_argument('--tasks_per_round', type=int,
                        default=5,
                        help="Number of sub-tasks (batch-size)")
    parser.add_argument('--epochs', type=int,
                        default=100,
                        help="Number of Epochs")

    parser.add_argument('--data_config', type=str,
                        default="./config/dataset/coarse_cifar100.yaml",
                        help="Multi-task config")
    parser.add_argument('--replay_frac', type=float,
                        default=1.0,
                        help="Fraction of samples used for replay")

    parser.add_argument('--hp_config', type=str,
                        default="./config/hyperparam/wrn.yaml",
                        help="Hyper parameter configuration")

    args = parser.parse_args()
    data_conf = fetch_configs(args.data_config)
    hp_conf = fetch_configs(args.hp_config)
    args.fp16 = args.gpu = torch.cuda.is_available()
    args.model = hp_conf['model']
    args.dataset = data_conf['dataset']
    args.data = fetch_dataclass(data_conf['dataset'])

    # Choose best implementation for functions
    # Does sacrifice exact reproducability from random seed
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    download_dataset(args, data_conf)

    # Train Model Zoo
    zoo = ModelZoo(args, data_conf, hp_conf)
    zoo.train()


if __name__ == '__main__':
    main()
