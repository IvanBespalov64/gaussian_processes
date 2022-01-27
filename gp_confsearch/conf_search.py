import trieste
import gpflow
import numpy as np
import tensorflow as tf
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from mean_cos import BaseCos

# defines a functions, that will convert args for mean_function
def parse_args_to_mean_func(inp):
    """
    Convert input data from [inp_dim, 7] to [7, inp_dim]
    """
    return tf.transpose(inp)

np.random.seed(1793)
tf.random.set_seed(1793)

# defines a function that will be predicted
# cur - input data 'tensor' [n, inp_dim], n - num of points, inp_dim - num of dimensions
def func(cur):
    return tf.map_fn(fn = lambda x : np.array([sum([tf.sin(c * 10) / 5 for c in x])]), elems = cur)

search_space = Box([0, 0], [2, 2])  # define the search space directly

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# defining GPR model
kernel = gpflow.kernels.Periodic(gpflow.kernels.Matern12(), 0.2 * np.pi) # Setting a kernel for GP; Do not forget about period!!!
gpr = gpflow.models.GPR(initial_data.astuple(), kernel, BaseCos(2, *parse_args_to_mean_func(tf.constant([[1., 1., 1., 1., 1., 1., 1.], \
                                                                                                                  [1., 1., 2., 2., 3., 3., 1.]]))), noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)
model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 5
result = bo.optimize(num_steps, initial_data, model)
dataset = result.try_get_final_dataset()

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")
