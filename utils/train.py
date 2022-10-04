import torch
import numpy as np
import wandb
from tqdm import tqdm
from PIL import Image
import faiss
from scipy import stats

def fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config, start_epoch=0, metrics = []):
    
    # Constants
    device = config.device
    n_epochs = config.n_epochs
    model_id = config.model_id

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, test_loader, model, loss_fn, optimizer, device, epoch, model_id)
        scheduler.step()

        # print('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

        map5 = test_epoch(test_loader, model, device, epoch)

        #print('Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,val_loss))

        state_dict = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': {metric.name(): metric.value() for metric in metrics},
            'map5': map5,
            'train_loss': train_loss,
        }

        wandb.log({"train_loss": train_loss, "map5": map5})

        PATH = 'models/' + model_id + '.pth'
        torch.save(state_dict, PATH)

def train_epoch(train_loader, test_loader, model, loss_fn, optimizer, device, epoch_num, model_id):
    model.train()
    model.to(device)
    loop = tqdm(train_loader, desc=f"EPOCH {epoch_num} TRAIN", leave=True)
    total_loss = 0
    map = -1
    for idx, (data, _) in enumerate(loop):

        if len(loop) > 2000 and (idx % 1000 == 0):
            map = test_epoch(test_loader, model, device, epoch_num)
            state_dict = {
                'epoch': epoch_num + 1,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            wandb.log({'map': map})

            PATH = f'models/{model_id}' + '.pth'
            torch.save(state_dict, PATH)
            model.train()

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
            log = {
                **log,
                'epoch': epoch_num,
                'iter': idx,
                'train_loss': loss.item(),
            }
            wandb.log(log)
        loop.set_postfix(loss=total_loss / (idx+1), map=round(map, 4))

    metrics = {}
    return total_loss, metrics

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=(self.k + 1))
        self.predictions = self.y[indices][:, 1:] # Discard the sample itself
        # predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return self.predictions

    def mapk(self):
        # TODO could be faster if it was implemented entirely with numpy
        map = [np.count_nonzero(self.predictions[idx, :] == l) for idx, l in enumerate(self.y)]
        return sum(map) / (len(map) * self.k)

def test_epoch(loader, model, device, epoch_num):
    model.eval()
    model.to(device)

    # hard-coded as the challenge stipulates 64 dimensions
    test_embeddings = np.empty((0, 64))
    test_labels = np.empty((0))
    path_list = []

    # loop = tqdm(loader, desc=f"EPOCH {epoch_num}  TEST", leave=True)
    with torch.no_grad():
        for idx, (data, target, path) in enumerate(loader):
            data = data.to(device)

            outputs = model.embedding(data)
            test_embeddings = np.vstack((test_embeddings, outputs.cpu().detach().numpy()))
            test_labels = np.concatenate((test_labels, target.numpy()))
            path_list += list(path)

    KNN = FaissKNeighbors(k=5)
    KNN.fit(X=test_embeddings, y=test_labels)
    pred = KNN.predict(X=test_embeddings)
    map = KNN.mapk()

    img = Image.open(path_list[0]).resize((224, 224))
    wandb.log({"query": wandb.Image(img)})

    ret = pred[0,:]
    ret_path = [path_list[int(p)] for p in ret]
    wandb.log({"retrievals": [wandb.Image(Image.open(p).resize((224, 224))) for p in ret_path]})

    return map