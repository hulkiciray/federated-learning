import torch
import torch.nn as nn
from flwr.server import criterion
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from datasets import Dataset
from flwr.client import NumPyClient
import flwr as fl
from tqdm import tqdm
from typing import List
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Load dataset from text file
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    return data

def preprocess_data(dataset, tokenizer):
    """This function preprocess data and create dataloaders for train and validation sets."""

    text_list = []
    for text in dataset:
        if text != "\n":
            text_list.append(text[:-1])

    # ####################TRAIN DATA####################
    inputs = tokenizer(text_list[:500], return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    inputs["labels"] = inputs.input_ids.detach().clone()
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []
    for i in range(mask_arr.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(mask_arr.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    train_data = CustomDataset(inputs)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # ####################TEST DATA####################
    val_inputs = tokenizer(text_list[500:800], return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    val_inputs["labels"] = val_inputs.input_ids.detach().clone()
    val_rand = torch.rand(val_inputs.input_ids.shape)
    val_mask_arr = (val_rand < 0.15) * (val_inputs.input_ids != 101) * (val_inputs.input_ids != 102) * (val_inputs.input_ids != 0)

    val_selection = []
    for i in range(val_mask_arr.shape[0]):
        val_selection.append(
            torch.flatten(val_mask_arr[i].nonzero()).tolist()
        )

    for i in range(val_mask_arr.shape[0]):
        val_inputs.input_ids[i, val_selection[i]] = 103

    val_data = CustomDataset(val_inputs)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

    return train_dataloader, val_dataloader

def train(model, dataloader, epochs:int, learning_rate=2e-5, verbose=False, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # MLM training detail ###
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model, loss

def test(model, dataloader, device="cpu"):
    """Evaluate the model on the validation set."""
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss += criterion(outputs, labels).item()

    return loss, loss / len(dataloader)

def set_parameters(model, parameters: List[np.ndarray]):
    """This is a helper function to update local model parameters with the global ones."""
    params_dict = zip(model.state_dict().keys(), parameters) # `state_dict to access PyTorch model parameter tensors
    # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict}) ## Flower.ai sitedeki örnek kodda yer alan kısım
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    """This is another helper function to get the updated model parameters from the local model."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, epochs, device, verbose):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.verbose = verbose

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters):
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, self.val_loader, verbose=self.verbose, epochs=self.epochs)
        return get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters):
        set_parameters(self.model, parameters)
        loss, acc = test(self.model, self.val_loader, device=self.device)
        return loss, len(self.val_loader), {"accuracy": float(acc)}

config = {
    "data_path": "data/somali.txt",
    "device": ("mps:0" if torch.backends.mps.is_available() else "cpu"),
    "model": BertForMaskedLM.from_pretrained("bert-base-multilingual-cased"),
    "tokenizer": BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    "optimizer": AdamW,
    "learning_rate": 5e-5,
    "epochs": 2,
    "verbose": True
}

if __name__ == "__main__":

    #device = config["device"]
    path = config["data_path"]

    #model = config["model"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    data = load_dataset(file_path=path)
    train_loader, val_loader = preprocess_data(dataset=data, tokenizer=tokenizer)

    # Create and start client
    client = FlowerClient(model=config["model"],
                          train_loader=train_loader,
                          val_loader=val_loader,
                          epochs=config["epochs"],
                          device=config["device"],
                          verbose=config["verbose"])

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)