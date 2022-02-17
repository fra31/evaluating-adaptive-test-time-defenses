import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import numpy as np
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
import argparse
import tqdm
from time import sleep

from Linear_control_funs import Linear_Control
from model import resnet20
import utils


class ModelRequestHandler(BaseHTTPRequestHandler):
  predict_batch_fn = None
  get_whitebox_data_fn = None
  shutdown_flag = False

  def _set_headers(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()

  def do_GET(self):
    if self.path == "/ping":
      response = "pong"
    elif self.path == "/get_whitebox_data":
      response = ModelRequestHandler.get_whitebox_data_fn()
    elif self.path == "/__shutdown__":
      response = "Shutting down."
      ModelRequestHandler.shutdown_flag = True
    else:
      response = ValueError(f"Unknown GET path {self.path}!")
    self._set_headers()
    self.wfile.write(pickle.dumps(response))

  def do_HEAD(self):
    self._set_headers()

  def do_POST(self):
    if self.path == "/predict_batch":
      content_length = int(self.headers['Content-Length'])
      post_data = self.rfile.read(content_length)
      request = pickle.loads(post_data)

      if not isinstance(request, np.ndarray):
        response = ValueError("Wrong data type(expected ndarray): ", type(request))
      else:
        response = ModelRequestHandler.predict_batch_fn(request)
    else:
      response = ValueError(f"Unknown POST path {self.path}!")
    self._set_headers()
    self.wfile.write(pickle.dumps(response))

  def log_message(self, format, *args):
    return        # Silence HTTP log output


def compute_projection_matrices(model, train_loader, max_num_samples, device):
  """ Compute the linear projection matrices as described in the paper.
      This needs a lot of RAM, so it can be helpful to run this on CPU.
  """
  with torch.no_grad():
    Lin = Linear_Control(model, max_num_samples=max_num_samples)
    Lin.compute_Princ_basis(train_loader, device=device)
    Lin.from_basis_projection()
  return Lin.Proj


def main():
  print("Model server starting.")
  utils.fix_random_seed()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Model server will run with device {device}.")

  # Only for learning the projection matrices. Control the evaluation batch size
  # in evaluate.py
  batch_size = 128
  workers = 2

  # These hyperparams control the strength of the defense. They are rather low,
  # but it's the only config that retains reasonable clean accuracy (89%).
  pmp_lr = 0.005
  pmp_max_iter = 5

  max_num_samples_for_proj = 5000        # As mentioned in the paper

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_loader = torch.utils.data.DataLoader(
  datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, 4),
      transforms.ToTensor(),
      normalize,
    ]), download=True),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

  model = resnet20(num_classes=10)
  model = model.to(device)
  model.load_state_dict(torch.load('models_cifar10/resnet20_model.ckpt', map_location=device))
  model.eval()

  # Compute the projections on CPU because it needs huge amounts of RAM.
  # Afterwards, upload the result to GPU again and inject into the model.
  print("Learning linear projections, this could take a while (10min+)...")
  model = model.to("cpu")
  proj = compute_projection_matrices(model, train_loader, max_num_samples=max_num_samples_for_proj, device="cpu")
  model = model.to(device)
  print("Finished learning linear projections.")
  Lin = Linear_Control(model, max_num_samples=max_num_samples_for_proj)
  Lin.Proj = proj
  for i in Lin.PCA_INDEX:
    Lin.Proj[i] = Lin.Proj[i].to(device).detach()

  def predict_batch(x: np.ndarray) -> np.ndarray:
    # Treat input & output as ndarray so they can be pickled
    assert np.min(x) >= 0.0
    assert np.max(x) <= 1.0
    x = torch.Tensor(x).to(device)
    x = normalize(x)
    outputs = Lin.PMP(x, learning_rate=pmp_lr, radius=0., max_iterations=pmp_max_iter)
    return outputs.detach().cpu().numpy()

  def get_whitebox_data() -> np.ndarray:
    # In the white-box setting, we are allowed to inspect the defender and
    # get the control parameters that were used for the last prediction
    return [cont.detach().cpu().numpy() for cont in Lin.last_Conts]

  http_host = "localhost"
  http_port = 6969
  ModelRequestHandler.predict_batch_fn = predict_batch
  ModelRequestHandler.get_whitebox_data_fn = get_whitebox_data
  httpd = HTTPServer((http_host, http_port), ModelRequestHandler)
  print(f"Model server is ready. Serving HTTP requests on {http_host}:{http_port}...")
  while not ModelRequestHandler.shutdown_flag:
    httpd.handle_request()

  print("Model server terminated.")

if __name__ == '__main__':
  main()
