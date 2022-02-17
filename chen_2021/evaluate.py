import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import numpy as np
import random
import argparse
import subprocess
import tqdm
from time import sleep
from autoattack import AutoAttack

from Linear_control_funs import Linear_Control
from model import resnet20
from model_client import ProxyModel
import utils


def clean_accuracy(test_loader, proxy_model: ProxyModel):
  # Report accuracy of the defense (remote) on clean images.
  test_loss = 0
  correct = 0
  total = 0
  for i, (inputs, labels) in enumerate(tqdm.tqdm(test_loader)):
    inputs = inputs.detach().numpy()
    labels = labels.detach().numpy()
    outputs = proxy_model.predict_batch(inputs)
    predicted = np.argmax(outputs, 1)
    total += len(inputs)
    correct += (predicted == labels).sum().item()
  return correct / total

def clean_accuracy_undef(test_loader, undef_model, device):
  # Report accuracy of the undefended model (locally) on clean images.
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  test_loss = 0
  correct = 0
  total = 0
  for i, (inputs, labels) in enumerate(tqdm.tqdm(test_loader)):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = undef_model(normalize(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  return correct / total


def robust_accuracy_pgd(test_loader, proxy_model, undefended_model, eps, batch_size, device, use_bpda=True):
  # Report accuracy against a 20-step PGD attack.
  # If use_bpda=False, then the attack is conducted against the static model,
  # and the adversarial examples are tested agaist (transferred to) the defense.
  # If use_bpda=True, then the attack is directly conducted against the defense,
  # using BPDA to approximate its gradients.

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # Build a copy of the defense
  surrogate_Lin = Linear_Control(undefended_model)

  def bpda_forward(x):
    """ Evaluating this function will return the predictions of the proxy model
        Calling backward() on it will return gradients of a surrogate. """
    # Get prediction from the defended model
    defense_preds = proxy_model.predict_batch(x.detach().cpu().numpy())
    defense_preds = torch.Tensor(defense_preds).to(device)

    # Get the control parameters that were used for the prediction
    last_Conts = proxy_model.get_whitebox_data()
    last_Conts = [torch.Tensor(c).to(device) for c in last_Conts]

    # Using the control parameters, we can build a network that matches the one
    # built by the defender for this prediction. Code is copypasted from the
    # Linear_Control.PMP() function, so the forward pass matches the defender's
    # for this particular x. The backward pass won't be the the same, but it
    # turns out it's a very good approximation.
    data_prop = normalize(x)
    for jj in range(len(surrogate_Lin.PCA_INDEX)):
      data_Conted = data_prop + last_Conts[jj]
      if jj != (len(surrogate_Lin.PCA_INDEX)-1):
        data_prop = surrogate_Lin.seq_model[surrogate_Lin.PCA_INDEX[jj]:surrogate_Lin.PCA_INDEX[jj+1]](data_Conted)
    surrogate_preds = surrogate_Lin.seq_model[surrogate_Lin.PCA_INDEX[-1]:](data_Conted)

    # surrogate_preds should equal defense_preds, but there might be some
    # numerical effects. Return the original predictions just to be sure.
    surrogate_preds.data = defense_preds
    return surrogate_preds

  def transfer_forward(x):
    return undefended_model(normalize(x))

  # Parameters same as in paper's original PGD eval code
  step_size = eps / 4.
  print("step size:", step_size)
  n_steps = 20
  ce_loss_fn = torch.nn.CrossEntropyLoss()

  correct = 0
  total = 0
  pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
  for i, (inputs, labels) in pbar:
    inputs, labels = inputs.to(device), labels.to(device)

    x_adv = inputs
    for i_step in range(n_steps):
      x_adv = x_adv.clone().detach().requires_grad_(True)
      prediction = bpda_forward(x_adv) if use_bpda else transfer_forward(x_adv)

      _, clsids = torch.max(prediction.data, 1)
      n_correct = (clsids == labels).sum().item()
      # print(f"PGD it {i_step}, batch acc:", n_correct/len(inputs))

      loss = ce_loss_fn(prediction, labels)
      loss.backward()

      with torch.no_grad():
        adv_gradient = x_adv.grad
        adv_gradient = adv_gradient.sign()

        adv_gradient *= step_size
        x_adv += adv_gradient

        x_adv = torch.maximum(torch.minimum(x_adv, inputs + eps), inputs - eps)
        x_adv = torch.clamp(x_adv, 0., 1.)

      # print("dist:", torch.max(torch.abs(x_adv - inputs)))

    # Double-check adversarial examples against the defended model.
    # Also display robust accuracy so far in the progress bar
    outputs = proxy_model.predict_batch(x_adv.detach().cpu().numpy())
    predicted = np.argmax(outputs, 1)
    total += len(inputs)
    correct += (predicted == labels.detach().cpu().numpy()).sum().item()
    pbar.set_description(f"acc={correct/total}")
  return correct / total


def robust_accuracy_aa(test_loader, proxy_model, undefended_model, eps, batch_size, device):
  # Report accuracy against APGD+BPDA

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # Build a copy of the defense
  surrogate_Lin = Linear_Control(undefended_model)

  def bpda_forward(x):
    """ Evaluating this function will return the predictions of the proxy model
        Calling backward() on it will return gradients of a surrogate. """
    # Get prediction from the defended model
    defense_preds = proxy_model.predict_batch(x.detach().cpu().numpy())
    defense_preds = torch.Tensor(defense_preds).to(device)

    # Get the control parameters that were used for the prediction
    last_Conts = proxy_model.get_whitebox_data()
    last_Conts = [torch.Tensor(c).to(device) for c in last_Conts]

    # Using the control parameters, we can build a network that matches the one
    # built by the defender for this prediction. Code is copypasted from the
    # Linear_Control.PMP() function, so the forward pass matches the defender's
    # for this particular x. The backward pass won't be the the same, but it
    # turns out it's a very good approximation.
    data_prop = normalize(x)
    for jj in range(len(surrogate_Lin.PCA_INDEX)):
      data_Conted = data_prop + last_Conts[jj]
      if jj != (len(surrogate_Lin.PCA_INDEX)-1):
        data_prop = surrogate_Lin.seq_model[surrogate_Lin.PCA_INDEX[jj]:surrogate_Lin.PCA_INDEX[jj+1]](data_Conted)
    surrogate_preds = surrogate_Lin.seq_model[surrogate_Lin.PCA_INDEX[-1]:](data_Conted)

    # surrogate_preds should equal defense_preds, but there might be some
    # numerical effects. Return the original predictions just to be sure.
    surrogate_preds.data = defense_preds
    return surrogate_preds

  auto_attack = AutoAttack(bpda_forward, norm='Linf', eps=eps, verbose=True, seed=999)
  auto_attack.attacks_to_run = ['apgd-ce']#, 'apgd-dlr', 'apgd-t'] #, 'fab-t']
  auto_attack.apgd.verbose = True
  auto_attack.apgd_targeted.verbose = True

  correct = 0
  total = 0
  pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
  for i, (inputs, labels) in pbar:
    inputs, labels = inputs.to(device), labels.to(device)
    adv_ex = auto_attack.run_standard_evaluation(inputs, labels, bs=batch_size)

    # Double-check adversarial examples against the defended model.
    # Also display robust accuracy so far in the progress bar
    outputs = proxy_model.predict_batch(adv_ex.detach().cpu().numpy())
    predicted = np.argmax(outputs, 1)
    total += len(inputs)
    correct += (predicted == labels.detach().cpu().numpy()).sum().item()
    pbar.set_description(f"acc={correct/total}")
  return correct / total


def main() -> None:
  utils.fix_random_seed()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Eval script will run with device {device}.")

  batch_size = 16    # max batch size that works with 4GB GPU RAM
  num_workers = 2
  eps = 2. / 255.

  cifar10_1k = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # Don't normalize here - conduct attacks with data in [0,1] and
        # normalize inside the forward fn.
    ]))
  cifar10_1k = data_utils.Subset(cifar10_1k, torch.arange(1000))
  test_loader = torch.utils.data.DataLoader(cifar10_1k,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)

  undefended_model = resnet20(num_classes=10)
  undefended_model = undefended_model.to(device)
  undefended_model.load_state_dict(torch.load('models_cifar10/resnet20_model.ckpt', map_location=device))
  undefended_model.eval()

  print("Evaluating accuracy of undefended model on clean data...")
  acc_clean = clean_accuracy_undef(test_loader, undefended_model, device)
  print(f"Accuracy on clean data: {acc_clean}")

  # Will block until the model server has built the model (could be very long)
  defended_model_proxy = ProxyModel("localhost", 6969)

  print("Evaluating accuracy of defense on clean data...")
  acc_clean = clean_accuracy(test_loader, defended_model_proxy)
  print(f"Accuracy on clean data: {acc_clean}")

  print("Running PGD20...")
  acc_pgd = robust_accuracy_pgd(test_loader,
                               proxy_model=defended_model_proxy,
                               undefended_model=undefended_model,
                               eps=eps,
                               batch_size=batch_size,
                               device=device)
  print(f"Robust accuracy against PGD20: {acc_pgd}")

  print("Running AutoAttack...")
  acc_robust = robust_accuracy_aa(test_loader,
                                  proxy_model=defended_model_proxy,
                                  undefended_model=undefended_model,
                                  eps=eps,
                                  batch_size=batch_size,
                                  device=device)
  print('Robust accuracy against AutoAttack:', acc_robust)

  print("Sending shutdown command to model server.")
  defended_model_proxy.shutdown_server()


if __name__ == '__main__':
  main()
