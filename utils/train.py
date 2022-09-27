import torch
import numpy as np
import wandb
from tqdm import tqdm
import faiss
from scipy import stats

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
        # train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, epoch)
        #
        # print('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

        val_loss, metrics = test_epoch(test_loader, model, loss_fn, device, epoch)
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


class FaissKNNImpl:

    def __init__(self, k, faiss):
        self.k = k  # k nearest neighbor value
        self.faissIns = faiss  # FAISS instance
        self.index = 0
        self.gpu_index_flat = 0
        self.train_labels = []
        self.test_label_faiss_output = []

    def fitModel(self, train_features, train_labels):
        self.train_labels = train_labels
        self.index = self.faissIns.IndexFlatL2(train_features.shape[1])  # build the index
        self.index.add(train_features)  # add vectors to the index

    def fitModel_GPU(self, train_features, train_labels):
        no_of_gpus = self.faissIns.get_num_gpus()
        self.train_labels = train_labels
        self.gpu_index_flat = self.index = self.faissIns.IndexFlatL2(train_features.shape[1])  # build the index
        if no_of_gpus > 0:
            self.gpu_index_flat = self.faissIns.index_cpu_to_all_gpus(self.index)

        self.gpu_index_flat.add(train_features)  # add vectors to the index
        return no_of_gpus

    def predict(self, test_features):
        distance, test_features_faiss_Index = self.index.search(test_features, self.k)
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index], axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        # for test_index in range(0,test_features.shape[0]):
        #    self.test_label_faiss_output[test_index] = stats.mode(self.train_labels[test_features_faiss_Index[test_index]])[0][0] #Counter(self.train_labels[test_features_faiss_Index[test_index]]).most_common(1)[0][0]
        return self.test_label_faiss_output

    def predict_GPU(self, test_features):
        distance, test_features_faiss_Index = self.gpu_index_flat.search(test_features, self.k)
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index], axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        return self.test_label_faiss_output

    def getAccuracy(self, test_labels):
        accuracy = (self.test_label_faiss_output == test_labels).mean()
        return round(accuracy, 2)

def test_epoch(loader, model, loss_fn, device, epoch_num):

    model.to(device)

    # hard-coded as the challenge stipulates 64 dimensions
    test_embeddings = np.empty((0, 64))
    test_labels = np.empty((0))

    loop = tqdm(loader, desc=f"EPOCH {epoch_num}  TEST", leave=True)
    with torch.no_grad():
        for idx, (data, target) in enumerate(loop):
            data = data.to(device)

            outputs = model.embedding(data)
            test_embeddings = np.vstack((test_embeddings, outputs.cpu().detach().numpy()))
            test_labels = np.concatenate((test_labels, target.numpy()))

            if idx == 3:
                break

    index = faiss.IndexFlatL2(64)

    KNN = FaissKNNImpl(k=5, faiss=index)
    n = KNN.fitModel_GPU(train_features=test_embeddings, train_labels=test_labels)
    print(n)
    n = KNN.predict(test_features=test_embeddings)
    print(n)

    print("all embeddings computed")
