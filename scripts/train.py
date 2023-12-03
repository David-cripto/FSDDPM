from fsddpm.methods.model import get_model
from torch.optim import Adam
from torch.utils.data import DataLoader
from fsddpm.datasets.colored_mnist import get_dataset
import torch
from tqdm import trange
from fsddpm.methods.EI.train import loss_fn, VPSDE

TIME_EMB_TYPE = "fourier"
DEVICE = "cpu"

LR = 1e-4
N_EPOCHS = 50
IMG_SIZE = 28
BATCH_SIZE = 32

def main():
    model = get_model(img_size = IMG_SIZE, time_embedding_type = TIME_EMB_TYPE)
    model.to(DEVICE)
    model.train()
    sde = VPSDE()

    optimizer = Adam(model.parameters(), lr=LR)

    train_dataset, _ = get_dataset("./")
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tqdm_epoch = trange(N_EPOCHS)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(DEVICE)    
            loss = loss_fn(sde, model, x)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(model.state_dict(), 'ckpt.pth')


if __name__ == '__main__':
    main()