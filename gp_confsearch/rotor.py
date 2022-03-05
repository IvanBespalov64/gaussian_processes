from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import math

def change_dihedral(mol_file_name : str, a1 : int, a2 : int, a3 : int, a4 : int, rad : float) -> str:
    """
        changing dihedral in radians between a1, a2, a3 and a4
        atoms (numeration starts with zero),
        returns coords block of xyz file
    """
    mol = Chem.MolFromMolFile(mol_file_name)
    rdMolTransforms.SetDihedralRad(mol.GetConformer(), a1, a2, a3, a4, rad)
    return '\n'.join(Chem.MolToXYZBlock(mol).split('\n')[2:])
    
