import os, json
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from .models.cnn import CNN

parser = ArgumentParser(description="An Evolutionary Stochastic Gradient Descent Trainer")
parser.add_argument("scheme", choices=["baseline", "esgd", "esgd_ws"])
parser.add_argument("--model", type=str, default="cnn")
parser.add_argument("-a", dest="data_augmentation", action="store_true")
parser.add_argument("--dataset", type=str, default="mnist")
args = parser.parse_args()

DATA_DIR = os.path.expanduser("~/esgd-ws/datasets")
# WEIGHTS_DIR = "./model_weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DICT = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST
}
MODEL_DICT = {"cnn": CNN}
DATA_FOLDER = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST"}

results_dir = os.path.expanduser(f"~/esgd-ws/results/{args.dataset}/{'DA' if args.data_augmentation else 'Non-DA'}")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

transform = None \
    if not args.data_augmentation \
    else transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(
            size=(28, 28),
            scale=(0.8, 1),
            ratio=(3 / 4, 4 / 3)
        ),
        transforms.ToTensor(),
    ])
download = not os.path.exists(os.path.join(DATA_DIR, DATA_FOLDER[args.dataset]))

if args.scheme == "baseline":
    # overwrite the transformation for baseline
    transform = transforms.ToTensor() \
        if not args.data_augmentation \
        else transforms.Compose([
            transforms.RandomResizedCrop(
                size=(28, 28),
                scale=(0.8, 1),
                ratio=(3 / 4, 4 / 3)
            ),
            transforms.ToTensor(),
        ])

    train_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=True, transform=transform, download=download)
    test_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=False, transform=transform, download=download)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = MODEL_DICT[args.model]()
    model.to(DEVICE)

    EPOCHS = 100
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for e in range(EPOCHS):
        with tqdm(train_loader, desc=f"{e + 1}/{EPOCHS} epochs") as t:
            running_correct = 0
            running_loss = 0
            running_total = 0
            model.train()
            for i, (x, y) in enumerate(t):
                out = model(x.to(DEVICE))
                loss = loss_fn(out, y.to(DEVICE))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = out.max(dim=1)[1].detach()
                running_correct += (pred == y.to(DEVICE)).sum().item()
                running_loss += loss.item() * x.size(0)
                running_total += x.size(0)
                if i < len(train_loader) - 1:
                    t.set_postfix({
                        "train_loss": running_loss / running_total,
                        "train_acc": running_correct / running_total
                    })
                else:
                    test_running_correct = 0
                    test_running_loss = 0
                    test_running_total = 0
                    model.eval()
                    with torch.no_grad():
                        for x, y in test_loader:
                            out = model(x.to(DEVICE))
                            test_running_loss += loss_fn(out, y.to(DEVICE)).item() * x.size(0)
                            test_running_correct += (out.max(dim=1)[1] == y.to(DEVICE)).sum().item()
                            test_running_total += x.size(0)
                        t.set_postfix({
                            "train_loss": running_loss / running_total,
                            "train_acc": running_correct / running_total,
                            "test_loss": test_running_loss / test_running_total,
                            "test_acc": test_running_correct / test_running_total,
                        })
                        train_losses.append(running_loss / running_total)
                        train_accuracies.append(running_correct / running_total)
                        test_losses.append(test_running_loss / test_running_total)
                        test_accuracies.append(test_running_correct / test_running_total)
                    scheduler.step()
    # torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"{args.scheme}_{MODEL_DICT[args.model]}.pt"))

    pd.DataFrame({
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }).to_csv(f"{results_dir}/{args.scheme}.csv")

if args.scheme == "esgd":
    from .esgd import ESGD, get_current_time

    train_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=True, download=download)
    train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    train_targets = torch.LongTensor(train_set.targets)
    test_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=False, download=download)
    test_set = torch.FloatTensor(test_set.data / 255).unsqueeze(1), torch.LongTensor(test_set.targets)

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }

    esgd = ESGD(
        hpset=HPSET,
        model_class=MODEL_DICT[args.model],
        n_generations=100
    )
    results = esgd.train(
        train_data,
        train_targets,
        test_set=test_set,
        batch_size=1024,
        transform=transform
    )
    with open(f"{results_dir}/{args.scheme}.json", "w") as f:
        json.dump(results, f)

if args.scheme == "esgd_ws":
    from .esgd_ws import ESGD_WS, get_current_time

    train_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=True, download=download)
    train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    train_targets = torch.LongTensor(train_set.targets)
    test_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=False, download=download)
    test_set = torch.FloatTensor(test_set.data / 255).unsqueeze(1), torch.LongTensor(test_set.targets)

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }

    esgd_ws = ESGD_WS(
        hpset=HPSET,
        model_class=MODEL_DICT[args.model],
        n_generations=100
    )
    results = esgd_ws.train(
        train_data,
        train_targets,
        test_set=test_set,
        batch_size=1024,
        transform=transform
    )
    with open(f"{results_dir}/{args.scheme}.json", "w") as f:
        json.dump(results, f)
