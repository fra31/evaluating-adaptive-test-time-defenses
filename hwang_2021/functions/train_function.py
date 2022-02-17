import numpy as np
import torch
import torch.optim as optim
from foolbox import PyTorchModel
from tqdm import tqdm

from functions import base_function
from functions import defense_function


def model_train(discriminator, target_model, trainloader, device, args):

    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    fmodel = PyTorchModel(target_model, bounds=(0, 1))

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        adv_data = base_function.generate_pgd_for_training(fmodel, inputs, targets, args)

        x_nat = inputs
        x = adv_data

        perturb = (x - x_nat) * args.gamma
        adversarial_vertex = x_nat + perturb
        adversarial_vertex = torch.clamp(adversarial_vertex, 0.0, 1.0)

        y_nat = 1
        y_vertex = 0

        x_weight = torch.from_numpy(np.random.beta(1.0, 1.0, [x.shape[0], 1, 1, 1])).type(torch.FloatTensor).to('cuda')

        y_weight = x_weight.reshape((-1, 1))

        x = x_nat * x_weight + adversarial_vertex * (1 - x_weight)
        y = y_nat * y_weight + y_vertex * (1 - y_weight)

        optimizer.zero_grad()

        outputs = discriminator(x, x, args)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()


def model_test(discriminator, target_model, testloader, attack, device, args):
    attacked_correct = 0
    defenced_correct = 0
    total = 0

    fmodel = PyTorchModel(target_model, bounds=(0, 1))

    print('')
    print('Start evaluating on ' + attack)
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_data = base_function.generate_attack_images(args.dataset, attack, fmodel, target_model, inputs, targets)

        with torch.no_grad():
            attacked_outputs = target_model(adv_data)

        defenced_data = defense_function.pgd_defense(discriminator=discriminator, images=adv_data, args=args)
        with torch.no_grad():
            defenced_outputs = target_model(defenced_data)

        total += targets.size(0)

        _, predicted = attacked_outputs.max(1)
        attacked_correct += predicted.eq(targets).sum().item()

        _, predicted = defenced_outputs.max(1)
        defenced_correct += predicted.eq(targets).sum().item()

    print('Attacked Acc: %.3f%% (%d/%d)' % (100. * attacked_correct / total, attacked_correct, total))
    print('Defensed Acc: %.3f%% (%d/%d)' % (100. * defenced_correct / total, defenced_correct, total))
    return 100. * attacked_correct / total, 100. * defenced_correct / total

