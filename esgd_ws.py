import torch
import functools
import numpy as np
from esgd import ESGD,get_current_time


class ESGD_WS(ESGD):

    def _sample_optimizer(self, weights=None):
        hpval_indices = np.random.choice(len(self.hpvalues), size=self.n_population, p=weights)
        return [self.hpvalues[idx] for idx in hpval_indices], hpval_indices

    def _update_weights(self, weights, rank, topn=3):
        for idx in rank[:topn]:
            weights[idx] += 1/np.sqrt(self.n_generations)
        return weights/np.sum(weights)

    def train(
            self,
            train_data,
            train_targets,
            topn=3,
            init_weights=None,
            test_set=None,
            log_file=None
    ):
        logger = self.Logger(log_file)
        if init_weights is None:
            weights = np.ones(len(self.hpvalues))
        else:
            weights = np.array(init_weights)
        train_loader = self.get_data_loader(train_data, train_targets)
        if test_set is not None:
            test_loader = self.get_data_loader(*test_set, shuffle=False)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        curr_gen = [self.model_class().to(self.device) for _ in range(self.n_population)]
        results = []
        for g in range(1, 1 + self.n_generations):
            curr_hpvals,hpval_indices = self._sample_optimizer()
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
            opt_rank = hpval_indices[np.argsort(running_losses)]
            weights = self._update_weights(weights, rank=opt_rank, topn=topn)
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
                    model = self.model_class().to(self.device)
                    for p_child, *p_parents in zip(model.parameters(), *[curr_gen[idx].parameters() for idx in mix]):
                        p_child.data = functools.reduce(lambda x, y: x + y, p_parents) / self.mixing_number
                        p_child.data.add_(1 / g * self.mutation_length * (2 * torch.rand_like(p_child) - 1))
                    offsprings.append(model)
            train_losses = [0.0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
            train_corrects = [0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
            if test_set is not None:
                test_losses = [0.0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
                test_corrects = [0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
                test_total = 0
            curr_gen.extend(offsprings)
            with torch.no_grad():
                for ind in curr_gen:
                    ind.eval()
                for (x, y) in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    for i, ind in enumerate(curr_gen):
                        out = ind(x)
                        train_losses[i] += self.fitness_function(out, y).item() * x.size(0)
                        train_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
                if test_set is not None:
                    for (x, y) in test_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        test_total += x.size(0)
                        for i, ind in enumerate(curr_gen):
                            out = ind(x)
                            test_losses[i] += self.fitness_function(out, y).item() * x.size(0)
                            test_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
                for ind in curr_gen:
                    ind.train()
            train_losses = list(map(lambda x: x / running_total, train_losses))
            train_accs = list(map(lambda x: x / running_total, train_corrects))
            if test_set is not None:
                test_losses = list(map(lambda x: x / test_total, test_losses))
                test_accs = list(map(lambda x: x / test_total, test_corrects))
            curr_rank = np.argsort(train_losses)
            elite = curr_rank[:self.m_elite]
            others = np.random.choice(len(curr_gen) - self.m_elite,
                                      size=self.n_population - self.m_elite) + self.m_elite
            others = curr_rank[others]
            curr_gen = [curr_gen[idx] for idx in np.concatenate([elite, others])]
            train_losses = [train_losses[idx] for idx in np.concatenate([elite, others])]
            train_accs = [train_accs[idx] for idx in np.concatenate([elite, others])]
            if test_set is not None:
                test_losses = [test_losses[idx] for idx in np.concatenate([elite, others])]
                test_accs = [test_accs[idx] for idx in np.concatenate([elite, others])]
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpost-EVO")
                logger.logging(f"\t|___population best fitness: {min(train_losses)}")
                logger.logging(f"\t|___population average fitness: {sum(train_losses) / len(train_losses)}")
                logger.logging(f"\t|___population best accuracy: {max(train_accs)}")
                logger.logging(f"\t|___population average accuracy: {sum(train_accs) / len(train_accs)}")
                if test_set is not None:
                    logger.logging(f"\t|___(test) population best test fitness: {min(test_losses)}")
                    logger.logging(f"\t|___(test) population average test fitness: {sum(test_losses) / len(test_losses)}")
                    logger.logging(f"\t|___(test) population best accuracy: {max(test_accs)}")
                    logger.logging(f"\t|___(test) population average test accuracy: {sum(test_accs) / len(test_accs)}")
            results.append({
                "train_losses": train_losses,
                "train_accs": train_accs,
                "test_losses": test_losses,
                "test_accs": test_accs
            })
        return results


if __name__ == "__main__":
    import os
    from models.cnn import CNN
    from torchvision import datasets, transforms

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
    LOG_DIR = "./log/esgd_ws"
    if not os.path.exists(LOG_DIR):
        os.mkdirs(LOG_DIR)

    esgd_ws = ESGD_WS(
        hpset=HPSET,
        model_class=CNN
    )
    esgd_ws.train(
        train_data,
        train_targets,
        test_set=test_set,
        log_file=f"{LOG_DIR}/{get_current_time()}.log"
    )
