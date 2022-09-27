import os
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.parser import load_yml
from utils.models import Triplet_CLIP_MLP
from utils.dataset import TripletGUIE
from utils.losses import TripletLoss
from utils.train import fit
import wandb

def main():

    cfg = load_yml("config.yml")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Device: {cfg.device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    OUTPUT_MODEL_DIR = './models/'
    device = cfg.device

    # Create the output directory if it does not exist
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    model = Triplet_CLIP_MLP(clip_model='ViT-B-32', pretrained='openai') # Joan: Fixed triplet model
    model.to(device)

    train_datasets = cfg.train_datasets
    test_datasets = cfg.test_datasets

    train_dataset = TripletGUIE(root=cfg.dataset_root,
                                train=True,
                                datasets=train_datasets)

    test_dataset = TripletGUIE(root=cfg.dataset_root,
                               train=False,
                               datasets=test_datasets)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=cfg.num_workers)

    criterion = TripletLoss(margin=1.)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = lr_scheduler.StepLR(opt, 8, gamma=0.1, last_epoch=-1)

    start_epoch = 0
    # Check if file exists
    if os.path.exists(OUTPUT_MODEL_DIR + cfg.model_id + '.pth'):
        print('Loading the model from the disk')
        checkpoint = torch.load(OUTPUT_MODEL_DIR + cfg.model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # model.load_state_dict(torch.load(OUTPUT_MODEL_DIR + model_id + '.pth'))

    print('Starting training, EPOCH: ', start_epoch)

    # Configure wandb logger
    wandb.init(
            project=cfg.project_name,
            entity=cfg.wandb_entity,
            name = cfg.model_id,
            resume=False,
            config=cfg,
        )

    fit(train_loader=train_loader, test_loader = test_loader, model=model, loss_fn=criterion, optimizer=opt, scheduler = scheduler, config=cfg, start_epoch=0)

if __name__ == "__main__":
    main()
