import torch
import torch.nn as nn

torch.manual_seed(0)

def joint_criterion(model, aux_criterion, X, y, alpha=1.):
    aux_loss, l = aux_criterion(model, X, joint=True, train=True)
    if aux_criterion.__name__ == 'recon_criterion':
        y = y
    elif aux_criterion.__name__ == 'pi_criterion':
        y = y.repeat(2)
    elif aux_criterion.__name__ == 'rotate_criterion':
        y = y.repeat(4)
    loss = nn.functional.cross_entropy(l, y)
    
    return loss + aux_loss * alpha, (loss, aux_loss)

def recon_criterion(model, X, joint=False, train=False, reduction='mean'):
    l = model(X, add_noise=train)
    loss = nn.functional.mse_loss(model.r, X, reduction=reduction)
    if not joint:
        return loss
    return loss, l

def pi_criterion(model, X, joint=False, train=False, reduction='mean'):
    if not train:
        X1 = X
    else:
        X1 = random_trans_combo(X)
    X2 = random_trans_combo(X, df=~joint)
    l1, l2 = model(X1), model(X2)
    l = torch.cat((l1, l2), dim=0)
    loss = nn.functional.mse_loss(l1, l2, reduction=reduction)
    if not joint:
        return loss
    return loss, l

def random_trans_combo(tensor, df=False):
    # tensor: bs * c * h * w
    if not df:
        tensor += (torch.randn_like(tensor)*0.1).clamp(0,1)
    if torch.rand(1) > 0.5 or df:
        tensor = tensor.flip(3)
    if not df:
        r_h = torch.randint(0, 8, (1,)).item()
        r_w = torch.randint(0, 8, (1,)).item()
        h = torch.randint(24, 32-r_h, (1,))
        w = torch.randint(24, 32-r_w, (1,))
    else:
        r_h, r_w, h, w = 2, 2, 28, 28
    tensor = tensor[:, :, r_h:r_h+h, r_w:r_w+w]
    return nn.functional.interpolate(tensor, [32, 32])

def rotate_criterion(model, X, joint=False, train=False, reduction='mean'):
    X_rotated = []
    for deg in [0, 90, 180, 270]:
        X_rotated.append(rotate_images(X, deg))
    X = torch.cat(X_rotated, dim=0)
    l = model(X, add_noise=train)
    y_deg = torch.arange(4)[:, None].repeat(1, X.shape[0]//4).flatten().to(X.device)
    loss = nn.functional.cross_entropy(model.pred_deg, y_deg, reduction=reduction)
    if not joint:
        if reduction == 'none':
            return loss.view(4, -1).mean(dim=0)
        return loss
    return loss, l

def rotate_criterion_l2(model, X, joint=False, train=False, reduction='mean'):
    X_rotated = []
    for deg in [0, 90, 180, 270]:
        X_rotated.append(rotate_images(X, deg))
    X = torch.cat(X_rotated, dim=0)
    l = model(X, add_noise=train)
    # l = model(X)
    y_deg = torch.arange(4)[:, None].repeat(1, X.shape[0]//4).flatten().to(X.device)
    onehot = torch.zeros(X.shape[0], 4).to(X.device)
    onehot[torch.arange(X.shape[0]), y_deg] = 1
    loss = nn.functional.mse_loss(torch.softmax(model.pred_deg, dim=1), onehot, reduction=reduction)
    if not joint:
        if reduction == 'none':
            return loss.sum(dim=1, keepdim=True).view(4, -1, 1).sum(dim=0)
        return loss
    return loss, l

def rotate_images(X, degree=0):
    if degree == 0:
        return X
    elif degree == 90:
        return X.transpose(2,3).flip(3)
    elif degree == 180:
        return X.flip(2).flip(3)
    elif degree == 270:
        return X.transpose(2,3).flip(2)

def second_order(model, X, y, df_criterion, beta=1):
    pred = model(X)
    return nn.functional.cross_entropy(pred, y) - df_criterion(model, X) * beta * 100