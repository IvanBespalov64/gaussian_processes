import gpflow
import numpy as np

from mean_cos import BaseCos

def normalize(energies : list[list[float]]) -> np.ndarray:
    """
        Normalize energies by following expr:
            E' = (E - min(E)) * 627.509474063
        Returns np.ndarray with same shape 
    """

    casted = np.array(energies)
    
    return (casted - np.min(casted, axis=1).reshape(casted.shape[0], 1)) * 627.509474063

def get_mf_coefs(energies : np.ndarray, 
                 degrees : np.ndarray = np.linspace(0., 2 * np.pi, 37).reshape(37, 1)) -> np.ndarray:
    """
        Calculates coefs of mean function for
        given dependency of energy from degree. 
        By default degrees is [0., 2 * pi] with step 10
    """
    
    model = gpflow.models.GPR((degrees.astype('double'), 
                               energies.reshape(degrees.shape[0], 1).astype('double')), 
                               gpflow.kernels.Periodic(gpflow.kernels.Matern12(), 2 * np.pi), 
                               BaseSin())
        
    gpflow.optimizers.Scipy().minimize(
       model.training_loss,
       variables=model.trainable_variables)

    return np.array([param.numpy()[0] for param in model.mean_function.parameters])

def get_coef_matrix(all_energies)

print(normalize([[1., 2., 3.,], [2., 7., 8.]])) 
