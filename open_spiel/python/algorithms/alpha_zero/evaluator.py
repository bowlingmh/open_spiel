# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An MCTS Evaluator for an AlphaZero model."""

import numpy as np

from open_spiel.python.algorithms import mcts
import pyspiel
from open_spiel.python.utils import lru_cache
from open_spiel.python.utils import spawn
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from multiprocessing import Pipe

def batch_inference_runner(*, config, queue):
  batch_size = config.inference_batch_size
  assert batch_size > 0

  model = model_lib.init_model_from_config(config)

  conns = []
  batch = []

  while True:
    msg = queue.get()

    if not msg: # Empty message ends the process
      return
    elif type(msg) == str: # Load a new model checkpoint
      model.load_checkpoint(msg)
    else: # Pipe containing an query for inference
      conn = msg
      query = conn.recv()

      conns.append(conn)
      batch.append(query)

    # Process a batch
    if len(conns) >= batch_size:
      obs = np.stack(b[0] for b in batch)
      mask = np.stack(b[1] for b in batch)

      value, policy = model.inference(obs, mask)

      for i,conn in enumerate(conns):
        conn.send((value[i,0], policy[i]))

      conns = []
      batch = []
      

class AlphaZeroEvaluator(mcts.Evaluator):
  """An AlphaZero MCTS Evaluator."""

  def __init__(self, config,
               inference_queue=None,
               logger=None,
               cache_size=2**16):
    """An AlphaZero MCTS Evaluator."""
    self._logger = logger
    self._cache = lru_cache.LRUCache(cache_size)
    self._inference_queue = inference_queue

    if self._inference_queue:
      self._inference_pipe = Pipe()
      self._model = None
    else:
      self._model = model_lib.init_model_from_config(config)

  def cache_info(self):
    return self._cache.info()

  def clear_cache(self):
    self._cache.clear()

  def load_model_checkpoint(self, path):
    if self._logger:
      self._logger.print("Inference cache:", self.cache_info())
      self._logger.print("Loading checkpoint", path)

    if self._model:
      self._model.load_checkpoint(path)

    self.clear_cache()

  def _inference(self, state):
    obs = np.array(state.observation_tensor())
    mask = np.array(state.legal_actions_mask())

    # ndarray isn't hashable
    cache_key = obs.tobytes() + mask.tobytes()

    rv = self._cache.get(cache_key)
    if rv: return rv

    if self._model:
      value, policy = self._model.inference(np.expand_dims(obs, 0),
                                            np.expand_dims(mask, 0))
      rv = value[0, 0], policy[0]  # Unpack batch
    else:
      # Put the query into the batching queue and wait for response
      self._inference_queue.put(self._inference_pipe[1])
      self._inference_pipe[0].send((obs, mask))
      rv = self._inference_pipe[0].recv()

    self._cache.set(cache_key, rv)
    return rv

  def evaluate(self, state):
    """Returns a value for the given state."""
    value, _ = self._inference(state)
    return np.array([value, -value])

  def prior(self, state):
    """Returns the probabilities for all actions."""
    _, policy = self._inference(state)
    return [(action, policy[action]) for action in state.legal_actions()]
