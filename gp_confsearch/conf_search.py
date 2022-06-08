#import calc # For init CURRENT_STRUCTURE_ID

from calc import calc_energy
from calc import change_dihedrals
from calc import parse_points_from_trj

from coef_calc import CoefCalculator

from sparse_ei import SparseExpectedImprovement

import trieste
import gpflow
import numpy as np
import tensorflow as tf
from trieste.data import Dataset
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from mean_cos import BaseCos
import plotly.graph_objects as go
from scipy.special import erf
import sys

from rdkit import Chem

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

MOL_FILE_NAME = "tests/cur.mol"
NORM_ENERGY = 0.
RANDOM_DISPLACEMENT = True

DIHEDRAL_IDS = []

CUR_ADD_POINTS = []

# defines a functions, that will convert args for mean_function
def parse_args_to_mean_func(inp):
    """
    Convert input data from [inp_dim, 7] to [7, inp_dim]
    """
    return tf.transpose(inp)

#np.random.seed(1793)
#tf.random.set_seed(1793)

def temp_calc(a : float, b  :float) -> float:
    """
        fast temp calc with cur dihedrals
    """
    if(tf.is_tensor(a)):
        a = a.numpy()
    if(tf.is_tensor(b)):
        b = b.numpy() 
    return (calc_energy(MOL_FILE_NAME, [([1, 2, 3, 4], a),\
                                         ([0, 1, 2, 3], b)], RANDOM_DISPLACEMENT) - NORM_ENERGY) * 627.509474063 

def calc(dihedrals : list[float]) -> float:
    """
        Perofrms calculating of energy with current dihedral angels
    """
    
    if tf.is_tensor(dihedrals):
        dihedrals = list(dihedrals.numpy())

    #print(list(zip(DIHEDRAL_IDS, dihedrals)))

    en, points = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, RANDOM_DISPLACEMENT)

    CUR_ADD_POINTS.append(points)

    return en
    #return (calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), RANDOM_DISPLACEMENT) - NORM_ENERGY) * 627.509474063

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
           

def save_plot(plot_name : str, points : list, z : list, plotBorder = True):
    """
        saves plot with points
    """
    print("Enter")
    fig = go.Figure()
    x, y = [], []
    z = z.reshape(len(z), )
    print("Init")
    for cur in points:
        a, b = cur
        x.append(a)
        y.append(b)
    print(len(x), len(y), len(z))
    fig = go.Figure(data=[go.Scatter3d(x = x, y = y, z = z,
                                   mode='markers',
                                   text = [_ for _ in range(1, len(points) + 1)])])
    fig.write_html(plot_name)
    if plotBorder:
        r = np.linspace(0, 2 * np.pi, 100)
        border = np.ones((100, 100)) * (np.min(z) + 3.)
        fig.add_trace(go.Surface(x = r, y = r, z = border))
        fig.write_html("tests/bordered.html")
    

def save_prob(file_name : str, model : gpflow.models.gpr.GPR, points : list, vals = None):
    """
        Saves max prob and plots last GP
    """
    fig = go.Figure()
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    predicted_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T
    mean, var = model.predict_f(predicted_points)
    #print(mean)
    prob = 0.5 * (erf((np.min(mean) + 3. - mean) / ((2 ** 0.5) * (var ** 0.5))) + 1)
    #print(prob)
    #print(f"Min prob: {np.min(prob)}\nMax prob: {np.max(prob)}")
    max_unknown_prob = 0.
    plot_points = []
    plot_prob = []
    for i in range(len(mean)):
        cx, cy = predicted_points[i]
        single = True
        for x, y in points:
            if(abs(cx - x) <= np.pi / 12 and abs(cy - y) <= np.pi / 12):
                single = False
                break
        if single:
            max_unknown_prob = max(max_unknown_prob, prob[i])
            plot_prob.append(prob[i])
        else:
            plot_prob.append([0.])
        plot_points.append(predicted_points[i])
    print(f"Max prob in unknown space: {max_unknown_prob}")
    fig.add_trace(go.Surface(x = xx, y = yy, z = np.array(plot_prob).reshape((30, 30)))) # plot_prob
    fig.add_scatter3d(x = xx, y = yy, z = vals, mode = "markers")
    print(vals)
    fig.write_html(file_name)

def get_prob(model : gpflow.models.gpr.GPR, 
             points : list, 
             vals : list[float]) -> np.ndarray:
    """
        Returns porb
    """  

    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    #predicted_points = np.vstack()
    
    pass

def get_max_unknown_prob(model : gpflow.models.gpr.GPR, points : list, vals : list, step = -1):
    """
        Returns max prob to find another minimum in unknow space
    """
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    predicted_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T
    mean, var = model.predict_f(predicted_points)
    prob = 0.5 * (erf((np.min(mean) + 3. - mean) / ((2 ** 0.5) * (var ** 0.5))) + 1)
    max_unknown_prob = 0.
    #print(points)
    for i in range(len(mean)):
        cx, cy = predicted_points[i]
        single = True
        for x, y in points:
            if(abs(cx - x) <= np.pi / 12 and abs(cy - y) <= np.pi / 12):
                single = False
                break
        if single:
            max_unknown_prob = max(max_unknown_prob, prob[i])
    print("Plotting prob!")
    save_prob(f"probs/prob_{step}.html", model, points, vals)
    #print("Plotting points!")
    #save_plot(f"probs/plot_{step}.html", points, val, False)
    print("Plotted!")
    return max_unknown_prob


# defines a function that will be predicted
# cur - input data 'tensor' [n, inp_dim], n - num of points, inp_dim - num of dimensions
def func(cur):
    
    return tf.map_fn(fn = lambda x : np.array([calc(x)]), elems = cur)
    #return tf.map_fn(fn = lambda x : np.array([sum([tf.sin(c * 10) / 5 for c in x])]), elems = cur)

def upd_points(dataset : Dataset, model : gpflow.models.gpr.GPR) -> tuple[Dataset, gpflow.models.gpr.GPR]:
    """
        update dataset and model from CUR_ADD_POINTS
    """

    degrees, energies = [], []
    for cur in CUR_ADD_POINTS:
        d, e = zip(*cur)
        degrees.extend(d)
        energies.extend(e)
    dataset += Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(energies), 1))
    model.update(dataset)
    model.optimize(dataset)

    return dataset, model

def upd_dataset_from_trj(trj_filename : str, dataset : Dataset) -> Dataset:
    """
        Return dataset that consists of old points
        add points from trj
    """
    degrees, energies = zip(*parse_points_from_trj(trj_filename, DIHEDRAL_IDS, NORM_ENERGY))

    return dataset + Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(degrees), 1))

coef_matrix = CoefCalculator(Chem.MolFromMolFile(MOL_FILE_NAME), "test_scans/").coef_matrix()

mean_func_coefs = []

for ids, coefs in coef_matrix:
    DIHEDRAL_IDS.append(ids)
    mean_func_coefs.append(coefs)

print(DIHEDRAL_IDS)
print(mean_func_coefs)

search_space = Box([0, 0], [2 * np.pi, 2 * np.pi])  # define the search space directly

#Calc normalizing energy
#in kcal/mol!
NORM_ENERGY, _ = calc_energy(MOL_FILE_NAME, [], 0, False, False)

print(NORM_ENERGY)

#CUR_ADD_POINTS.append(norm_points)

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
num_initial_points = 1
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

#print(initial_data)
#print(CUR_ADD_POINTS)

#try:
#    print(*parse_args_to_mean_func(\
#                                tf.constant([[-7.56389, 2.86674, 1.01007, -0.76272, 9.27928, 0.963203, 2.58037],
#                                    [-0.36729715, 0.89504444, -0.62781192, 1.50781726, 0.01071042, -1.98400616, 1.76391248]])))
#    print(*parse_args_to_mean_func(tf.constant(mean_func_coefs, dtype="float")))
#except Exception:
#    pass

# defining GPR model
kernel = gpflow.kernels.Periodic(gpflow.kernels.Matern12(), 2 * np.pi) # Setting a kernel for GP; Do not forget about period!!!
gpr = gpflow.models.GPR(initial_data.astuple(), kernel,\
                         BaseCos(2, *parse_args_to_mean_func(tf.constant(mean_func_coefs, dtype="float"))))
                                                                                
#                                 tf.constant([[-7.56389, 2.86674, 1.01007, -0.76272, 9.27928, 0.963203, 2.58037],
#                                    [-0.36729715, 0.89504444, -0.62781192, 1.50781726, 0.01071042, -1.98400616, 1.76391248]]))), noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)
model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

#rule = trieste.acquisition.rule.TrustRegion()

#num_steps = 5
#result = bo.optimize(num_steps, initial_data, model)
#dataset = result.try_get_final_dataset()

# First step
#result, history = bo.optimize(1, initial_data, model).astuple()

#result = bo.optimize(num_initial_points, initial_data, model)

#model = result.try_get_final_model()
#dataset = result.try_get_final_dataset()

dataset, model = upd_points(initial_data, model)

rule = EfficientGlobalOptimization(SparseExpectedImprovement(np.pi / 3))

#print(upd_dataset_from_trj("tests/cur_trj.xyz", dataset))

prev_result = None

#points = []
#obs = []

#dataset = result.unwrap().dataset

f = open("conf_search.log", "w+")

print(dataset, file=f)

probs = []

steps = 1
print("Begin opt!", file = f)
for _ in range(3):
    print(f"Step number {steps}")
    try:
        result = bo.optimize(1, dataset, model, rule, fit_initial_model = False)
        print(f"Optimization step {steps} succeed!", file = f)
    except Exception:
        print("Optimization failed", file = f)
        print(result.astuple()[1][-1].dataset, file = f)
    #result, new_history = bo.optimize(1, 
    #                              #history[-1].dataset, 
    #                              dataset,
    #                              history[-1].model).astuple()
    #                              #history[-1].acquisition_state).astuple()
    #history.extend(new_history)
    #dataset = result.unwrap().dataset
    #dataset += result.unwrap().dataset
    dataset = result.try_get_final_dataset()
    model = result.try_get_final_model()

    print(dataset)

    dataset = upd_dataset_from_trj("tests/cur_trj.xyz", dataset)
    model.update(dataset)
    model.optimize(dataset)

    print(dataset.query_points.numpy(), file = f)
    #points.extend(result.unwrap().dataset.query_points.numpy())
    #obs.extend(result.unwrap().dataset.observations.numpy())
    #print(points)    
    cur_prob = get_max_unknown_prob(model.model, dataset.query_points.numpy(), dataset.observations.numpy(), steps)
    save_plot(f"probs/plot_{steps}.html", dataset.query_points.numpy(), dataset.observations.numpy())
    #print(history[-1].dataset.query_points.numpy())
    print(f"Current prob: {cur_prob}", file = f)
    steps += 1
    prev_result = result
    #if(cur_prob <= 0.01):
    #    break
 
    probs.append(cur_prob)

#dataset = result.unwrap().dataset

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}", file = f)
print(f"observation: {observations[arg_min_idx, :]}", file = f)
save_res("tests/res.xyz", *query_points[arg_min_idx, :])
save_all("tests/all.xyz", query_points)

print(probs, file=f)

f.close()

with open("res.dat", "w+") as file:
    file.write(query_points.__str__())
    file.write("\n")
    file.write(observations.__str__())
save_plot("tests/plot.html", query_points, observations)
save_prob("tests/prob.html", result.try_get_final_model().model, query_points)
