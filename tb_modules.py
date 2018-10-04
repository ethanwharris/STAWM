from torchbearer.callbacks import add_to_loss


def kl_divergence(mu_key, logvar_key, beta=4):
    @add_to_loss
    def loss(state):
        mu = state[mu_key]
        logvar = state[logvar_key]

        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)

        return beta * total_kld.item()
    return loss
