import numpy as np

from utils.run_net import evaluate


class Logger():
    def __init__(self, test_loaders, train_loaders,
                 num_tasks, n_cls, args, tune):
        self.te_loaders = test_loaders
        self.tr_loaders = train_loaders
        self.gpu = args.gpu
        self.tasks = num_tasks
        self.n_cls = n_cls
        self.tune = tune

    def evaluate_train(self, net):
        self.train_accs = []
        for task in range(self.tasks):
            task_met = evaluate(net, self.tr_loaders[task], self.gpu)
            self.train_accs.append((task_met[0].item(), task_met[1].item()))

    def log_metrics(self, net, train_met, ep):
        """
        Log metrics and evaluation umbers
        """
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

        info = {
            "Epoch": ep+1, "TrainAcc": train_met[0],
            "TestAcc": test_met[0], "TrainLoss": train_met[1],
            "TestLoss": test_met[1], "AllTestMetrics (loss,acc)": task_accs,
            "AllTrainMetrics (loss,acc)": self.train_accs
        }
        if not self.tune:
            print(info)

        return test_met[0], test_met[1]

    def log_train(self, net, train_met, ep):
        """
        Log metrics and evaluation umbers
        """
        info = {
            "Epoch": ep+1, "TrainAcc": train_met[0],
            "TrainLoss": train_met[1]
        }
        print(info)
