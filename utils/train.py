import torch
import numpy as np
import wandb
from tqdm import tqdm

def fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config, start_epoch=0, metrics = []):
    
    # Constants
    device = config.device
    n_epochs = config.n_epochs
    model_id = config.model_id


    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, epoch)

        print('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

        val_loss, metrics = test_epoch(test_loader, model, loss_fn, device)
        val_loss /= len(test_loader)

        print('Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,val_loss))
    

        state_dict = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': {metric.name(): metric.value() for metric in metrics},
            'val_loss': val_loss,
            'train_loss': train_loss,
        }

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        PATH = 'models/' + model_id + '.pth'
        torch.save(state_dict, PATH)

def train_epoch(train_loader, model, loss_fn, optimizer, device, epoch_num):
    model.train()
    loop = tqdm(train_loader, desc=f"EPOCH {epoch_num} TRAIN", leave=True)
    total_loss = 0
    for idx, (data, _) in enumerate(loop):
        data = tuple(d.to(device) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        outputs = (outputs,) if type(outputs) not in (tuple, list) else outputs

        loss = loss_fn(*outputs)
        loss = loss[0] if type(loss) in (tuple, list) else loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        log = {}

        if idx % 10 == 0:
            print(epoch_num, idx, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch_num,
                'iter': idx,
                'train_loss': loss.item()
            }
            wandb.log(log)
        loop.set_postfix(loss=total_loss / (idx+1))

    metrics = {}  
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = tuple(t.cuda() for t in target)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # TO DO: Implement validation for the retrieval task and mAP@5
    metrics = {}
    return val_loss, metrics