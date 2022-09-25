import os
import os
import torch
import torchvision
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms as T
from torchvision import models
import urllib
import torch.optim as optim
from tqdm import tqdm

# def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch_num):
#     model.train()
#     loop = tqdm(loader, desc=f"EPOCH {epoch_num} TRAIN", leave=True)
#
#     correct = 0  # accumulated correct predictions
#     total_samples = 0  # accumulated total predictions
#     loss_sum = 0  # accumulated loss
#
#     for idx, (data, targets) in enumerate(loop):
#         data, targets = data.to(device), targets.to(device)  # data and labels to device
#         optimizer.zero_grad()  # Initialize gradients
#
#         outputs = model(data)  # Forward pass
#         loss = loss_fn(outputs, targets)  # Compute the loss
#         _, predictions = torch.max(
#             outputs.data, 1
#         )  # Obtain the classes with higher probability
#
#         total_samples += data.size(0)  # Subtotal of the predictions
#         correct += (
#             (predictions == targets).sum().item()
#         )  # Subtotal of the correct predictions
#         loss_sum += loss.item()  # Subtotal of the correct losses
#
#         scaler.scale(loss).backward()  # Backward pass
#         scaler.step(optimizer)  # Update the weights
#         scaler.update()  # Update the scale
#
#         loop.set_postfix(acc=correct / total_samples, loss=loss_sum / (idx + 1))
#
#     epoch_acc = correct / total_samples  # Epoch accuracy
#     epoch_loss = loss_sum / len(loader)  # Epoch loss
#
#     return epoch_acc, epoch_loss
#
#
# def eval_fn(loader, model, loss_fn, device, epoch_num):
#
#     model.eval()
#     loop = tqdm(
#         loader,  # Create the tqdm bar for visualizing the progress
#         desc=f"EPOCH {epoch_num}  TEST",
#         leave=True,
#     )
#
#     correct = 0  # Accumulated correct predictions
#     total_samples = 0  # Accumulated total predictions
#     loss_sum = 0  # Accumulated loss
#
#     with torch.no_grad():
#         for idx, (data, targets) in enumerate(loop):
#             data, targets = data.to(device), targets.to(
#                 device
#             )  # data and labels to device
#
#             outputs = model(data)  # Forward pass
#             loss = loss_fn(outputs, targets)  # Compute the loss
#             _, predictions = torch.max(
#                 outputs.data, 1
#             )  # Obtain the classes with higher probability
#
#             total_samples += data.size(0)  # subtotal of the predictions
#             correct += (
#                 (predictions == targets).sum().item()
#             )  # subtotal of the correct predictions
#             loss_sum += loss.item()  # Subtotal of the correct losses
#
#             loop.set_postfix(acc=correct / total_samples, loss=loss_sum / (idx + 1))
#
#     epoch_acc = correct / total_samples  # Epoch accuracy
#     epoch_loss = loss_sum / len(loader)  # Epoch loss
#
#     return epoch_acc, epoch_loss
#
#
# if __name__ == "__main__":
#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     print(f"PyTorch Version: {torch.__version__}")
#     print(f"Torchvision: {torchvision.__version__}")
#     print(f"Device: {DEVICE}")
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#
#     # Load ConvNeXt-Large model from torchvision trained in Imagenet
#     model = models.convnext_large(weights="IMAGENET1K_V1")
#
#     # Change the last layer to have an output of 64D and add an extra one
#     # to finetune the NN with CIFAR100.
#     # At the end, the last layer will be removed.
#     num_ftrs = model.classifier[-1].in_features
#     model.classifier[-1] = nn.Linear(num_ftrs, 64)
#     model.classifier.append(nn.Linear(64, 100))
#
#     print("Linear layers:")
#     print(model.classifier)
#     model = model.to(DEVICE)
#
#     train_dataset = torchvision.datasets.CIFAR100(
#         root=".",
#         train=True,
#         download=True,
#         transform=T.Compose(
#             [
#                 T.Resize((128, 128)),
#                 T.ToTensor(),
#             ]
#         ),
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=32, shuffle=True, num_workers=2
#     )
#
#     test_dataset = torchvision.datasets.CIFAR100(
#         root=".",
#         train=False,
#         download=True,
#         transform=T.Compose(
#             [
#                 T.Resize((128, 128)),
#                 T.ToTensor(),
#             ]
#         ),
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=32, shuffle=True, num_workers=2
#     )
#
#     print()
#     print("CIFAR100 dataset loaded")
#     print(f"Training images: {len(train_dataset)}")
#     print(f"Testing images: {len(test_dataset)}")
#
#     opt = optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     scaler = torch.cuda.amp.GradScaler()
#
#     train_metrics = {"accuracy": [], "loss": []}
#     test_metrics = {"accuracy": [], "loss": []}
#
#     for idx in range(50):
#
#         acc, loss = train_fn(
#             loader=train_loader,
#             model=model,
#             optimizer=opt,
#             loss_fn=criterion,
#             scaler=scaler,
#             device=DEVICE,
#             epoch_num=idx,
#         )
#
#         train_metrics["accuracy"].append(acc)
#         train_metrics["loss"].append(loss)
#
#         acc, loss = eval_fn(
#             loader=test_loader, model=model, loss_fn=criterion, device=DEVICE, epoch_num=idx
#         )
#
#         test_metrics["accuracy"].append(acc)
#         test_metrics["loss"].append(loss)
#
#         if acc == max(test_metrics["accuracy"]):
#             model.eval()
#             model_to_save = model
#             model_to_save.classifier = model_to_save.classifier[:-1]
#             saved_model = torch.jit.script(model_to_save)
#             saved_model.save("saved_model.pt")
#             print(f"Model saved at epoch {idx} with acc {acc}")
