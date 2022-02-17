import requests
import numpy as np
import pickle
from time import sleep


class ProxyModel:
    def __init__(self, address, port):
      self.address = address
      self.port = port
      self.base_url = f"http://{self.address}:{self.port}"
      self._block_until_connected()

    def _block_until_connected(self):
      connected = False
      while not connected:
        try:
          r = requests.get(f"{self.base_url}/ping")
          response = pickle.loads(r.content)
          connected = True
        except Exception as e:
          print("Waiting for model server to come up...")
          sleep(10)
      print("Connected to remote model server!")
      assert response == "pong", f"Got unexpected ping response: {response}"

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
      assert isinstance(x, np.ndarray)
      r = requests.post(f"{self.base_url}/predict_batch", data=pickle.dumps(x))
      response = pickle.loads(r.content)
      assert isinstance(response, np.ndarray), f"Unexpected response: {response}"
      return response

    def get_whitebox_data(self):
      r = requests.get(f"{self.base_url}/get_whitebox_data")
      response = pickle.loads(r.content)
      return response

    def shutdown_server(self):
      r = requests.get(f"{self.base_url}/__shutdown__")
