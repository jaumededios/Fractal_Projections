from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
def new_global_key():
  return jax.random.PRNGKey(np.random.randint(100000))


