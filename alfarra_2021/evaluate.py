# Code is mostly copypasted from the colab

import sys


def main() -> None:
  sys.path.insert(0,'Combating-Adversaries-with-Anti-Adversaries')
  sys.path.insert(0,'RayS')

  eps_linf = 0.031        # Almost 8/255. This is the exact value as used in the defense reference implementation
  n_imgs = 1000           # Don't eval more than n images
  batch_size = 64
  batch_size = min(batch_size, n_imgs)

  rays_n_queries = 10000

  # Get CIFAR10 dataset
  import numpy as np
  import torch
  import torchvision
  import torchvision.transforms as transforms

  transform = transforms.Compose(
      [transforms.ToTensor(),
       ])     # The model already contains the preprocessing

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=1)

  # Move everything into memory
  x_test_clean = torch.cat([x for (x, y) in testloader], 0).to("cuda")[:n_imgs]
  y_test = torch.cat([y for (x, y) in testloader], 0).to("cuda")[:n_imgs]

  # Get pretrained AWP model, same as used in the reference implementation
  from experiments.adv_weight_pert import get_model as get_awp_model
  model_undefended = get_awp_model(k=0, alpha=0).eval().to("cuda")
  model_defended = get_awp_model(k=2, alpha=0.15).eval().to("cuda")

  import tqdm

  def eval_acc(model, x, y_gt):
    assert x.shape[0] == y_gt.shape[0]
    n = x.shape[0]

    n_batches = n // batch_size
    if n % batch_size != 0:
      n_batches += 1

    correct = 0
    with torch.no_grad():
        for i_batch in tqdm.tqdm(range(n_batches)):
            excerpt = slice(i_batch * batch_size, (i_batch+1) * batch_size)
            outputs = model(x[excerpt])
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_gt[excerpt]).sum().item()
    return correct / n

  #print(f"Undefended model accuracy on clean imgs: {eval_acc(model_undefended, x=x_test_clean, y_gt=y_test)}")
  #print(f"Defended model accuracy on clean imgs: {eval_acc(model_defended, x=x_test_clean, y_gt=y_test)}")

  from general_torch_model import GeneralTorchModel
  from RayS import RayS

  def run_rays_attack(model, n_queries):
    print(f"Running RayS attack with {n_queries} queries. This could take a while...")
    rays_torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    attack = RayS(rays_torch_model, epsilon=eps_linf)

    n_batches = x_test_clean.shape[0] // batch_size
    if x_test_clean.shape[0] % batch_size != 0:
      n_batches += 1

    n_total = 0
    n_robust_correct = 0

    progress_bar = tqdm.tqdm(range(n_batches))
    for i_batch in progress_bar:
        excerpt = slice(i_batch * batch_size, (i_batch+1) * batch_size)
        x_batch_clean = x_test_clean[excerpt]
        y_batch_gt = y_test[excerpt]

        x_batch_adv, queries, adbd, succ = attack(data=x_batch_clean, label=y_batch_gt, query_limit=n_queries)
        #print(f"This batch: attack reports success rate of {torch.sum(succ).item() / x_batch_clean.shape[0]}")

        # Filter by attack success
        below_eps_filter = torch.max(torch.abs(x_batch_adv - x_batch_clean).view(x_batch_clean.shape[0], -1), dim=1)[0] < eps_linf
        if torch.sum(below_eps_filter) != torch.sum(succ):
          # Shouldn't happen, but if it does then it should be investigated
          print(f"WARN: Actual attack success ({torch.sum(below_eps_filter).item()}) != reported attack success ({torch.sum(succ).item()})!")

        # Combine clean images with successfully attacked images and measure overall accuracy
        x_batch_adv_below_eps = x_batch_clean.clone()
        x_batch_adv_below_eps[below_eps_filter] = x_batch_adv[below_eps_filter]
        outputs = model(x_batch_adv_below_eps)
        _, y_batch_pred = torch.max(outputs.data, 1)

        n_total += x_batch_clean.shape[0]
        n_robust_correct += (y_batch_pred == y_batch_gt).sum().item()
        robust_acc = n_robust_correct / n_total

        # This might take a long time, so display the running accuracy after every batch
        progress_bar.set_description(f"acc={robust_acc}")

    return robust_acc

  robust_acc = run_rays_attack(model_undefended, n_queries=rays_n_queries)
  print(f"Undefended model robust accuracy: {robust_acc}")

  robust_acc = run_rays_attack(model_defended, n_queries=rays_n_queries)
  print(f"Defended model robust accuracy: {robust_acc}")

  print("done")


if __name__ == '__main__':
  main()
