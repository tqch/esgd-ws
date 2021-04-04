import os, sys
from tqdm import tqdm
import torch
import torch.nn as nn

DATA_DIR = "./datasets"
WEIGHTS_DIR = "./model_weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if sys.argv[1] == "cnn":

    from models.cnn import CNN
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from torch.optim import SGD, lr_scheduler

    DOWNLOAD = not os.path.exists(os.path.join(DATA_DIR,"MNIST"))

    train_set = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=DOWNLOAD)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=DOWNLOAD)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = CNN()
    model.to(DEVICE)

    EPOCHS = 50
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8, nesterov=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

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
                    # scheduler.step()

    torch.save(model.state_dict(),os.path.join(WEIGHTS_DIR,"cnn_mnist.pt"))