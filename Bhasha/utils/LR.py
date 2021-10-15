import torch


def Scheduler(Params, optimizer, steps, epochs):
    if Params.lr_schedule == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=Params.lr_milestones, gamma=Params.lr_gamma)
    elif Params.lr_schedule == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=Params.lr_gamma)
    elif Params.lr_schedule == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=Params.lr_mode, patience=Params.lr_patience, factor=Params.lr_factor, threshold=Params.lr_threshold)
    elif Params.lr_schedule == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Params.lr_max, steps_per_epoch=steps, epochs=epochs)
    elif Params.lr_schedule == "CyclicLR":
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=Params.lr_base, max_lr=Params.lr_max)
    elif Params.lr_schedule == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=Params.lr_step_size, gamma=Params.lr_gamma)


def Optim(Params, model):
    if Params.optimizer == 'Adam':
        return torch.optim.Adam(
            model.parameters(), lr=Params.learning_rate, betas=Params.betas, eps=Params.eps, weight_decay=Params.weight_decay, amsgrad=Params.amsgrad)

    elif Params.optimizer == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(), lr=Params.learning_rate, betas=Params.betas, eps=Params.eps, weight_decay=Params.weight_decay, amsgrad=Params.amsgrad)

    elif Params.optimizer == 'SGD':
        return torch.optim.SGD(
            model.parameters(), lr=Params.learning_rate, momentum=Params.momentum, dampening=Params.dampening, weight_decay=Params.weight_decay, nesterov=Params.nesterov
        )

    elif Params.optimizer == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=Params.learning_rate, alpha=Params.alpha, eps=Params.eps,
                                   weight_decay=Params.weight_decay, momentum=Params.momentum, centered=Params.centered)

    elif Params.optimizer == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=Params.learning_rate,
                                  betas=Params.betas, eps=Params.eps, weight_decay=Params.weight_decay)

    elif Params.optimizer == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=Params.learning_rate, rho=Params.rho,
                                    eps=Params.eps, weight_decay=Params.weight_decay)

    elif Params.optimizer == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=Params.learning_rate, lr_decay=Params.lr_decay,
                                   weight_decay=Params.weight_decay,  eps=Params.eps)
