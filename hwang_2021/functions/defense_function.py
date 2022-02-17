import torch


def pgd_defense(discriminator, images, args, track_interm=False):

    images = images.detach()

    eps = args.defense_eps
    alpha = eps / 4
    iters = args.defense_step

    ori_images = images.clone().detach()

    interm_x = []
    
    for i in range(iters):
        images.requires_grad = True

        outputs = discriminator(images, images, args).sum()

        grad = torch.autograd.grad(outputs, images,
                                   retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()

        adv_images = images + alpha * grad_sign

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        del grad

        if track_interm:
            interm_x.append(images.clone())
    
    adv_images = images

    if not track_interm:
        return adv_images
    else:
        return (adv_images, interm_x)
