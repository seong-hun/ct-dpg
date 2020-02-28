import torch
import torch.nn as nn
import torch.optim as optim


def get_batch_model(in_size, out_size, hidden_sizes, bias=True):
    layers = []
    for hidden_size in hidden_sizes:
        layers += [
            nn.Linear(in_size, hidden_size, bias=bias),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2)
        ]
        in_size = hidden_size
    layers += [nn.Linear(in_size, out_size, bias=bias)]
    return nn.Sequential(*layers)


class HNet(nn.Module):
    def __init__(self, x_size, u_size):
        super().__init__()
        self.model = get_batch_model(x_size + u_size, 1, [64, 128, 64])

    def forward(self, x, u):
        xu = torch.cat((x, u), 1)
        out = self.model(xu)
        return out


class GradNet(nn.Module):
    def __init__(self, x_size, u_size):
        super().__init__()
        self.model = get_batch_model(x_size + u_size, u_size, [64, 128, 64])

    def forward(self, x, u):
        xu = torch.cat((x, u), 1)
        out = self.model(xu)
        return out


class PiNet(nn.Module):
    def __init__(self, x_size, u_size):
        super().__init__()
        self.model = get_batch_model(x_size, u_size, [], bias=False)

    def forward(self, x):
        out = self.model(x)
        return out

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)


class Agent(nn.Module):
    def __init__(self, x_size, u_size, reward_fn, reward_grad_fn,
                 lr_h=1e-3, lr_g=1e-3, lr_pi=1e-4):
        super().__init__()
        self.name = self.__class__.__name__
        self.net_h = HNet(x_size, u_size)
        self.net_g = GradNet(x_size, u_size)
        self.net_pi = PiNet(x_size, u_size)
        self.reward = self.numpy_wrapper(reward_fn)
        self.reward_grad = self.numpy_wrapper(reward_grad_fn)
        self.criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam([
            {"params": self.net_h.parameters(), "lr": lr_h},
            {"params": self.net_g.parameters(), "lr": lr_g}
        ])
        self.actor_optimizer = optim.Adam(
            self.net_pi.parameters(), lr=lr_pi)
        self.info = {}
        self.info["loss"] = {}

    def get_param(self):
        return self.net_pi.model[0].weight.detach().numpy()

    def numpy_wrapper(self, fn):
        def wrapper(x, u):
            x = x.detach().numpy()
            u = u.detach().numpy()
            res = torch.tensor(fn(x, u)).float()
            if res.ndim == 1:
                res = res[:, None]
            return res
        return wrapper

    def set_input(self, data):
        self.data = data

    def update(self):
        self.critic_optimizer.zero_grad()
        x, u = self.data
        h = self.net_h(x, u)
        with torch.no_grad():
            upi = self.net_pi(x)
        hpi = self.net_h(x, upi)
        g = self.net_g(x, u)
        pi = self.net_pi(x)
        R = self.reward(x, u) - self.reward(x, pi)
        delta_g = g - self.reward_grad(x, pi)
        loss = self.criterion(
            h + torch.einsum("bi,bi->b", delta_g, pi - u)[:, None], R)
        loss += self.criterion(hpi, torch.zeros_like(hpi)) * 1e-2
        loss.backward()
        self.info["loss"]["critic"] = loss.detach().numpy()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        with torch.no_grad():
            g = self.net_g(x, upi)
        loss = torch.sum(self.net_pi(x) * g, axis=1).mean()
        # loss += self.criterion(self.net_pi(x), u) * 1e1
        loss.backward()
        self.info["loss"]["actor"] = loss.detach().numpy()
        self.actor_optimizer.step()

    def get(self, t, x):
        x = torch.tensor(x).float()[None, :]
        u = self.net_pi(x)[0].detach().numpy()
        return u

    def get_name(self):
        return self.name

    def get_msg(self):
        loss_msg = [
            f"{name} loss: {loss: 5.4f}"
            for name, loss in self.info["loss"].items()]
        return "  ".join(loss_msg)

    def save(self, epoch, global_step, path):
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }, path)

    def load_weights(self, data):
        self.load_state_dict(data["state_dict"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
