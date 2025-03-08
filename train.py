import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_loss_func(train_losses, val_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

def train(model, train_loader, valid_loader, num_epochs, device, optimizer, criterion, plot_loss=False):
    train_losses = []
    val_losses = []
    model.to(device)

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        running_loss = 0.0
        logging.info(f"Epoch #{epoch + 1} out of {num_epochs} started")
        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        for images, labels in tqdm(valid_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    if plot_loss:
        plot_loss_func(train_losses, val_losses)