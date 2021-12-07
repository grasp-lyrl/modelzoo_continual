import torch
import torch.nn as nn
import numpy as np
import torch.cuda.amp as amp


def evaluate(net, dataloader, gpu):
    """
    Evaluate the network on a single task and return (acc, loss)
    """
    acc = 0.0
    loss = 0.0
    count = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        net.eval()
        for dat, target in dataloader:
            tasks, labels = target
            batch_size = int(labels.size()[0])
            labels = labels.long()
            tasks = tasks.long()

            if gpu:
                dat = dat.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                tasks = tasks.cuda(non_blocking=True)

            out = net(dat, tasks)
            loss += (criterion(out, labels).item()) * batch_size

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    ret = np.array((acc/count, loss/count))
    return ret


def run_epoch(net, args, optimizer, train_loader,
              lr_scheduler, scaler):
    """
    Train the model for one epoch
    """
    train_loss = 0.0
    train_acc = 0.0
    batches = 0.0
    criterion = nn.CrossEntropyLoss()

    net.train()
    for dat, target in train_loader:
        optimizer.zero_grad()

        tasks, labels = target
        labels = labels.long()
        tasks = tasks.long()
        batch_size = int(labels.size()[0])

        if args.gpu:
            dat = dat.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            tasks = tasks.cuda(non_blocking=True)

        # Forward/Back-prop
        with amp.autocast(enabled=args.fp16):
            out = net(dat, tasks)
            loss = criterion(out, labels)
            scaler.scale(loss).backward()

            # Update params
            scaler.step(optimizer)
            scaler.update()

        lr_scheduler.step()

        # Compute Train metrics
        batches += batch_size
        train_loss += loss.item() * batch_size
        labels = labels.cpu().numpy()
        out = out.cpu().detach().numpy()
        train_acc += np.sum(labels == (np.argmax(out, axis=1)))

    train_met = np.array((train_acc/batches, train_loss/batches))
    return train_met
