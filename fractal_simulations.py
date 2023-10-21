from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
def new_global_key():
  return jax.random.PRNGKey(np.random.randint(100000))


class Koch_Snowflake():
  def __init__(self,key,theta, n=10):
    self.theta = theta
    self.key = key
    self.n = n

  def affine_transforms(self,theta):

    def rotate(x,theta):
      x,y = x
      return jnp.array((x*jnp.cos(theta)-y*jnp.sin(theta), 
                       y*jnp.cos(theta)+x*jnp.sin(theta)))
    first = lambda x: jnp.array(x)
    second = lambda x: first((1,0))+rotate(x, theta)
    third = lambda x: second((1,0))+rotate(x,-theta)
    fourth = lambda x: third((1,0))+x
    return lambda x: jnp.array([first(x), second(x), 
                                third(x), fourth(x)])/(2+2*jnp.cos(theta))

  def dimension(self):
    theta = self.theta
    scale_ratio = 2+2*jnp.cos(theta)
    num_intervals = 4
    return jnp.log(num_intervals)/jnp.log(scale_ratio)


  # Returns a uniform random point in a random koch
  def random_point(self,key):
    koch_key = self.koch_key
    theta = self.theta
    n = self.n

    x = jnp.array((0,0))
    #to generate x, we make a list of what of the 4 intervals we pick at each level
    pos_choices = jax.random.randint(key, shape=(n,), minval=0, maxval=4) 

    #we generate the random fractal angles depending on the position
    #in a pretty ugly way
    fractal_angles = []
    for p in pos_choices:
      k,koch_key = jax.random.split(koch_key,2)
      koch_key = jax.random.PRNGKey(jax.random.randint(koch_key, shape=(), 
                                                       minval=0, maxval=10000)+p)
      fractal_choices.append(
          theta*(-1)**jax.random.randint(k, shape=(), minval=0, maxval=2) 
      )
    
    #generate the point by composing the transformations backwards
    for c,angle in zip(pos_choices[::-1],fractal_angles[::-1]):
      x = self.affine_transforms(angle)(x)[c]

    #center the fractal
    theta0 = fractal_angles[0]  
    x0 = jnp.array(( 1/2,jnp.sin(theta0)/(1+1*jnp.cos(theta0))/4))
    return x-x0


