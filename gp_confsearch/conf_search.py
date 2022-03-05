from calc import calc_energy
from calc import change_dihedrals
import trieste
import gpflow
import numpy as np
import tensorflow as tf
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from mean_cos import BaseCos
import plotly.graph_objects as go
from scipy.special import erf

MOL_FILE_NAME = "tests/cur.mol"
NORM_ENERGY = 0.

# defines a functions, that will convert args for mean_function
def parse_args_to_mean_func(inp):
    """
    Convert input data from [inp_dim, 7] to [7, inp_dim]
    """
    return tf.transpose(inp)

np.random.seed(1793)
tf.random.set_seed(1793)

def temp_calc(a : float, b  :float) -> float:
    """
        fast temp calc with cur dihedrals
    """
    if(tf.is_tensor(a)):
        a = a.numpy()
    if(tf.is_tensor(b)):
        b = b.numpy() 
    return (calc_energy(MOL_FILE_NAME, [([1, 2, 3, 4], a),\
                                         ([0, 1, 2, 3], b)]) - NORM_ENERGY) * 627.509474063 

def save_res(xyz_name : str, a : float, b : float):
    """
        saves to 'xyz_name' geometry with a and b
    """
    with open(xyz_name, 'w+') as file:
        file.write(change_dihedrals(MOL_FILE_NAME, [([1, 2, 3, 4], a), ([0, 1, 2, 3], b)], True))

def save_all(all_file_name : str, points : list):
    """
        saves all structures
    """
    with open(all_file_name, 'w+') as file:
        for cur in points:
	        a, b = cur
	        file.write(change_dihedrals(MOL_FILE_NAME, [([1, 2, 3, 4], a), ([0, 1, 2, 3], b)], True))
           

def save_plot(plot_name : str, points : list, z : list):
    """
        saves plot with points
    """
    fig = go.Figure()
    x, y = [], []
    z = z.reshape(len(z), )
    for cur in points:
        a, b = cur
        x.append(a)
        y.append(b)
    fig = go.Figure(data=[go.Scatter3d(x = x, y = y, z = z,
                                   mode='markers',
                                   text = [_ for _ in range(1, len(points) + 1)])])
    fig.write_html(plot_name)
    r = np.linspace(0, 2 * np.pi, 100)
    border = np.ones((100, 100)) * (np.min(z) + 3.)
    fig.add_trace(go.Surface(x = r, y = r, z = border))
    fig.write_html("tests/bordered.html")
    

def save_prob(file_name : str, model : gpflow.models.gpr.GPR, points : list):
    """
        Saves max prob and plots last GP
    """
    fig = go.Figure()
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    predicted_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T
    mean, var = model.predict_f(predicted_points)
    print(mean)
    prob = 0.5 * (erf((np.min(mean) + 3. - mean) / ((2 ** 0.5) * var)) + 1)
    print(prob)
    print(f"Min prob: {np.min(prob)}\nMax prob: {np.max(prob)}")
    max_unknown_prob = 0.
    plot_points = []
    plot_prob = []
    for i in range(len(mean)):
        cx, cy = predicted_points[i]
        single = True
        for x, y in points:
            if(abs(cx - x) <= np.pi / 6 and abs(cy - y) <= np.pi / 6):
                single = False
                break
        if single:
            max_unknown_prob = max(max_unknown_prob, prob[i])
            plot_prob.append(prob[i])
        else:
            plot_prob.append([0.])
        plot_points.append(predicted_points[i])
    print(f"Max prob in unknown space: {max_unknown_prob}")
    fig.add_trace(go.Surface(x = xx, y = yy, z = np.array(plot_prob).reshape((30, 30))))
    fig.write_html(file_name)

# defines a function that will be predicted
# cur - input data 'tensor' [n, inp_dim], n - num of points, inp_dim - num of dimensions
def func(cur):
    
    return tf.map_fn(fn = lambda x : np.array([temp_calc(*x)]), elems = cur)
    #return tf.map_fn(fn = lambda x : np.array([sum([tf.sin(c * 10) / 5 for c in x])]), elems = cur)

search_space = Box([0, 0], [2 * np.pi, 2 * np.pi])  # define the search space directly

#Calc normalizing energy
NORM_ENERGY = calc_energy(MOL_FILE_NAME)

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# defining GPR model
kernel = gpflow.kernels.Periodic(gpflow.kernels.Matern12(), 2 * np.pi) # Setting a kernel for GP; Do not forget about period!!!
gpr = gpflow.models.GPR(initial_data.astuple(), kernel,\
                         BaseCos(2, *parse_args_to_mean_func(\
                         tf.constant([[-7.56389, 2.86674, 1.01007, -0.76272, 9.27928, 0.963203, 2.58037],
                                      [-0.36729715, 0.89504444, -0.62781192, 1.50781726, 0.01071042, -1.98400616, 1.76391248]]))), noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)
model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 150
result = bo.optimize(num_steps, initial_data, model)
dataset = result.try_get_final_dataset()

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")
save_res("tests/res.xyz", *query_points[arg_min_idx, :])
save_all("tests/all.xyz", query_points)
with open("res.dat", "w+") as file:
    file.write(query_points.__str__())
    file.write("\n")
    file.write(observations.__str__())
save_plot("tests/plot.html", query_points, observations)
save_prob("tests/prob.html", result.try_get_final_model().model, query_points)
