import numpy as np
import logging
import json

from utils.run_net import evaluate


class Logger():
    def __init__(self, test_loaders, train_loaders,
                 num_tasks, n_cls, args):
        self.te_loaders = test_loaders
        self.tr_loaders = train_loaders
        self.gpu = args.gpu
        self.tasks = num_tasks
        self.n_cls = n_cls

    def evaluate_train(self, net):
        self.train_accs = []
        for task in range(self.tasks):
            task_met = evaluate(net, self.tr_loaders[task], self.gpu)
            self.train_accs.append((task_met[0].item(), task_met[1].item()))

    def log_metrics(self, net, train_met, ep, stdout=False):
        """
        Compute and loss train/test metrics. Slightly expensive
        so must be done infrequently during training (once every 10 epochs)
        """
        # Compute train/test metrics of learner.
        test_met = np.array([0.0, 0.0])
        self.evaluate_train(net)

        task_accs = []
        for task in range(self.tasks):
            task_met = evaluate(net, self.te_loaders[task], self.gpu)
            task_accs.append((task_met[0].item(), task_met[1].item()))
            test_met += task_met

        # Take average accuracy of all tasks
        # Doesn't give extra wt. to a task with more samples
        test_met = test_met / len(task_accs)

        def rnd(x):
            return [tuple(np.round(y, 3)) for y in x]

        info = {
            "Epoch": ep+1, "TrainAcc": train_met[0],
            "TestAcc": test_met[0], "TrainLoss": train_met[1],
            "TestLoss": test_met[1],
            "AllTestMetrics (acc,loss)": rnd(task_accs),
            "AllTrainMetrics (acc,loss)": rnd(self.train_accs)
        }
        logging.info(str(info))
        if stdout:
            taskacc = str(list(np.round(np.array(task_accs)[:, 0], 2)))
            print("Added learner with test accuracies: %s:" % taskacc)

        return test_met[0], test_met[1]

    def log_train(self, net, train_met, ep):
        """
        Log metrics obtained from training single epoch
        """
        info = {
            "Epoch": ep+1, "TrainAcc": train_met[0],
            "TrainLoss": train_met[1]
        }
        logging.info(json.dumps(info))
