import os, sys
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json


DATA_DIR = "./datasets"
WEIGHTS_DIR = "./model_weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if sys.argv[1] == "cnn_baseline":

    from models.cnn import CNN
    from torch.optim import SGD, lr_scheduler

    DOWNLOAD = not os.path.exists(os.path.join(DATA_DIR,"MNIST"))

    train_set = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=DOWNLOAD)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=DOWNLOAD)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = CNN()
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
                        train_losses.append(running_loss/running_total)
                        train_accuracies.append(running_correct / running_total)
                        test_losses.append(test_running_loss / test_running_total)
                        test_accuracies.append(test_running_correct / test_running_total)
                    # scheduler.step()
    torch.save(model.state_dict(),os.path.join(WEIGHTS_DIR,"cnn_mnist.pt"))

    RESULTS_DIR = "./results"
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    pd.DataFrame({
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }).to_csv(f"{RESULTS_DIR}/{sys.argv[1]}.csv")

if sys.argv[1] == "cnn_esgd":
    from models.cnn import CNN
    from esgd import ESGD,get_current_time

    DATA_DIR = "./datasets"
    DOWNLOAD = not os.path.exists(os.path.join(DATA_DIR, "MNIST"))

    train_set = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=DOWNLOAD)
    train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    train_targets = torch.LongTensor(train_set.targets)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=DOWNLOAD)
    test_set = torch.FloatTensor(test_set.data / 255).unsqueeze(1), torch.LongTensor(test_set.targets)

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }
    LOG_DIR = "./log"
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    esgd = ESGD(
        hpset=HPSET,
        model_class=CNN,
        n_generations=100
    )
    results = esgd.train(
        train_data,
        train_targets,
        test_set=test_set
    )
    json.dump(results, f"{RESULTS_DIR}/{sys.argv[1]}.json")

if sys.argv[1] == "cnn_esgd_ws":
    from models.cnn import CNN
    from esgd_ws import ESGD_WS,get_current_time

    DATA_DIR = "./datasets"
    DOWNLOAD = not os.path.exists(os.path.join(DATA_DIR, "MNIST"))

    train_set = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=DOWNLOAD)
    train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    train_targets = torch.LongTensor(train_set.targets)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=DOWNLOAD)
    test_set = torch.FloatTensor(test_set.data / 255).unsqueeze(1), torch.LongTensor(test_set.targets)

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }
    LOG_DIR = "./log"
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    esgd = ESGD_WS(
        hpset=HPSET,
        model_class=CNN,
        n_generations=100
    )
    results = esgd.train(
        train_data,
        train_targets,
        test_set=test_set,
        log_file=f"{LOG_DIR}/{get_current_time()}.log"
    )
    json.dump(results, f"{RESULTS_DIR}/{sys.argv[1]}.json")