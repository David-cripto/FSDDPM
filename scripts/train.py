from fsddpm.methods.model import get_model
from torch.optim import Adam
from torch.utils.data import DataLoader
from fsddpm.datasets.colored_mnist import get_dataset
import torch
from tqdm import trange
from fsddpm.methods.EI.train import loss_fn, VPSDE
from diffusers.optimization import get_scheduler

TIME_EMB_TYPE = "fourier"
DEVICE = "cuda"

LR = 4e-4
N_EPOCHS = 10**3
IMG_SIZE = 28
BATCH_SIZE = 128

def main():
    model = get_model(sample_size = IMG_SIZE, time_embedding_type = TIME_EMB_TYPE)
    model.to(DEVICE)
    model.train()
    sde = VPSDE()

    optimizer = Adam(model.parameters(), lr=LR)

    train_dataset, _ = get_dataset("./")
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_history = []

    lr_scheduler = get_scheduler(
        'constant_with_warmup',
        optimizer=optimizer,
        num_warmup_steps=1*len(data_loader),
        num_training_steps=N_EPOCHS*len(data_loader),
    )

    tqdm_epoch = trange(N_EPOCHS)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(DEVICE)    
            loss = loss_fn(sde, model, x)
            loss.backward()    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        loss_history.append(avg_loss / num_items)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': loss_history,
            }, 
            'ckpt.pth'
            )


if __name__ == '__main__':
    main()