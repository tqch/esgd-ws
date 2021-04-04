import os, sys, time
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import itertools
import functools
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_current_time():
    return datetime.fromtimestamp(time.time()).isoformat()


class ESGD:

    def __init__(
            self,
            hpset,
            model_class,
            fitness_function=nn.CrossEntropyLoss(),
            n_generations=10,
            n_population=5,
            sgds_per_gen=1,
            evos_per_gen=1,
            reproductive_factor=3,
            m_elite=3,
            backoff=False,
            mixing_number=3,
            optimizer_class=SGD,
            mutation_length_init=0.01,
            random_state=71,
            device=DEVICE,
            verbose=True
    ):
        self.hpnames, self.hpvalues = self.extract_from_hpset(hpset)
        self.model_class = model_class
        self.fitness_function = fitness_function
        self.n_generations = n_generations
        self.n_population = n_population
        self.sgds_per_gen = sgds_per_gen
        self.evos_per_gen = evos_per_gen
        self.reproductive_factor = reproductive_factor
        self.m_elite = m_elite
        self.mixing_number = mixing_number
        self.optimizer_class = optimizer_class
        self.mutation_length = mutation_length_init
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    @staticmethod
    def extract_from_hpset(hpset):
        hpnames = []
        hpvalues = []
        for k, v in hpset.items():
            hpnames.append(k)
            hpvalues.append(v)
        return tuple(hpnames), tuple(itertools.product(*hpvalues))

    @staticmethod
    def get_data_loader(train_data, train_targets, batch_size=1024):
        class Dataset:
            def __init__(self, train_data, train_targets):
                self.train_data = train_data
                self.train_targets = train_targets
                self.length = len(self.train_targets)

            def __getitem__(self, idx):
                return self.train_data[idx], self.train_targets[idx]

            def __len__(self):
                return self.length

        return DataLoader(Dataset(train_data, train_targets), batch_size=batch_size)

    class Logger:
        
        def __init__(self):
            self.log_file = None
            
        def logging(self, s):
            if self.log_file is None:
                sys.stdout.write(s)
            else:
                with open(self.log_file, "a+") as f:
                    f.write(s)

    def train(self, train_data, train_targets, log_file=None):
        logger = self.Logger(log_file)
        train_loader = self.get_data_loader(train_data, train_targets)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        curr_gen = [self.model_class().to(self.device) for _ in range(self.n_population)]
        for g in range(1, 1 + self.n_generations):
            curr_hpvals = np.random.choice(len(self.hpvalues), size=self.n_population)
            curr_hpvals = [self.hpvalues[idx] for idx in curr_hpvals]
            optimizers = [self.optimizer_class(
                ind.parameters(), **dict(zip(self.hpnames, hpvs))
            ) for ind, hpvs in zip(curr_gen, curr_hpvals)]
            running_losses = [0.0 for _ in range(self.n_population)]
            running_corrects = [0 for _ in range(self.n_population)]
            running_total = 0
            if self.verbose:
                logger.logging(f"Generation #{g}:")
                logger.logging(f"|___{get_current_time()}\tpre-SGD")
            for s in range(self.sgds_per_gen):
                for (x, y) in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    running_total += x.size(0)
                    for i, (ind, opt) in enumerate(zip(curr_gen, optimizers)):
                        out = ind(x)
                        loss = self.fitness_function(out, y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        running_losses[i] += loss.item() * x.size(0)
                        running_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
            running_losses = list(map(lambda x: x / running_total, running_losses))
            running_accs = list(map(lambda x: x / running_total, running_corrects))
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpost-SGD")
                logger.logging(f"\t|___population best fitness: {min(running_losses)}")
                logger.logging(f"\t|___population average fitness: {sum(running_losses) / len(running_losses)}")
                logger.logging(f"\t|___population best accuracy: {max(running_accs)}")
                logger.logging(f"\t|___population average accuracy: {sum(running_accs) / len(running_accs)}")
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpre-evolution")
            curr_mix = [
                np.random.choice(self.n_population, size=self.mixing_number)
                for _ in range(int(self.reproductive_factor * self.n_population))
            ]
            offsprings = []
            for e in range(self.evos_per_gen):
                for mix in curr_mix:
                    model = CNN().to(self.device)
                    for p_child, *p_parents in zip(model.parameters(), *[curr_gen[idx].parameters() for idx in mix]):
                        p_child.data = functools.reduce(lambda x, y: x + y, p_parents) / self.mixing_number
                        p_child.data.add_(1 / g * self.mutation_length * (2 * torch.rand_like(p_child) - 1))
                    offsprings.append(model)
            running_losses = [0.0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
            curr_gen.extend(offsprings)
            with torch.no_grad():
                for (x, y) in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    for i, ind in enumerate(curr_gen):
                        out = ind(x)
                        running_losses[i] += self.fitness_function(out, y).item() * x.size(0)
            running_losses = list(map(lambda x: x / running_total, running_losses))
            curr_rank = np.argsort(running_losses)
            elite = curr_rank[:self.m_elite]
            others = np.random.choice(len(curr_gen) - self.m_elite,
                                      size=self.n_population - self.m_elite) + self.m_elite
            others = curr_rank[others]
            curr_gen = [curr_gen[idx] for idx in np.concatenate([elite, others])]
            running_losses = [running_losses[idx] for idx in np.concatenate([elite, others])]
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpost-EVO")
                logger.logging(f"\t|___population best fitness: {min(running_losses)}")
                logger.logging(f"\t|___population average fitness: {sum(running_losses) / len(running_losses)}")


if __name__ == "__main__":
    from models.cnn import CNN
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from torch.optim import SGD
    from models.cnn import CNN

    DATA_DIR = "./datasets"
    DOWNLOAD = not os.path.exists(os.path.join(DATA_DIR, "MNIST"))

    train_set = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=DOWNLOAD)
    train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    train_targets = torch.LongTensor(train_set.targets)

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }
    LOG_DIR = "./log"

    esgd = ESGD(
        hpset=HPSET,
        model_class=CNN
    )
    esgd.train(train_data, train_targets, log_file=f"{get_current_time()}.log")
