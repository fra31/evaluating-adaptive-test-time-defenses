import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

from attacks import fgsm, pgd_linf, inject_noise

def defense_wrapper(model, criterion, X, defense, epsilon=None, step_size=None, num_iter=None,
    randomize=False, return_interm=False):
    
    model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta, int_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, step_size=step_size, num_iter=num_iter,
            randomize=randomize, return_interm=return_interm)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta, int_delta

def purify(model, #aux_criterion,
    X, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5,
    randomize=False, aux_criterion=None, return_interm=False):

    if aux_criterion is None:
        return X
    n_delta = 11 if delta > 0 else 1
    aux_track = torch.zeros(n_delta, X.shape[0])
    inv_track = torch.zeros(n_delta, *X.shape)
    int_track = torch.zeros(n_delta, *X.shape, num_iter)
    for e in range(n_delta): #11
        defense = partial(defense_wrapper, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter,
            randomize=randomize, return_interm=return_interm)
        inv_delta, int_delta = defense(model, X=X)
        inv_track[e] = inv_delta
        int_track[e] += int_delta.to(int_track.device)
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1)).detach()
    e_selected = aux_track.argmin(dim=0)
    int_delta = int_track[e_selected, torch.arange(X.shape[0])].to(X.device)
    #print(int_delta.shape)
    return (inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X,
        int_delta.permute([0, 4, 1, 2, 3]) + X.unsqueeze(1))