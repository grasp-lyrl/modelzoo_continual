import torch.nn as nn
import torch


class SmallConv(nn.Module):
    """
    Small convolution network with no residual connections
    """
    def __init__(self, num_task=1, num_cls=10, channels=3,
                 avg_pool=2, lin_size=320):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size

        lin_layers = []
        for task in range(num_task):
            lin_layers.append(nn.Linear(self.linsize, num_cls))

        self.fc = nn.ModuleList(lin_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, tasks):

        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.view(-1, self.linsize)

        logits = self.fc[0](x) * 0

        for idx, lin in enumerate(self.fc):
            task_idx = torch.nonzero((idx == tasks), as_tuple=False).view(-1)
            if len(task_idx) == 0:
                continue

            task_out = torch.index_select(x, dim=0, index=task_idx)
            task_logit = lin(task_out)
            logits.index_add_(0, task_idx, task_logit)

        return logits
