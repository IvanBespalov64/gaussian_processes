import os
import time
import math
from typing import Union
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

USE_ORCA = True
ORCA_EXEC_COMMAND = "/opt/orca5/orca"
GAUSSIAN_EXEC_COMMAND = "srung"
DEFAULT_NUM_OF_PROCS = 24
DEFAULT_METHOD = "RHF/STO-3G"
DEFAULT_ORCA_METHOD = "r2SCAN-3c TightSCF"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPL = 1


#Alias for type of node about dihedral angle 
#that consists of list with four atoms and value of degree
dihedral = tuple[list, float]

def change_dihedrals(mol_file_name : str,
                     dihedrals : list[dihedral], full_block = False) -> Union[str, None]:
    """
        changinga all dihedrals in 'dihedrals' 
        (degree in rads, numeration starts with zero),
        returns coords block of xyz file
        if no file exists, return None
        if fullBlock is True returns full xyz block
    """
    try:
        mol = Chem.MolFromMolFile(mol_file_name, removeHs = False)
        for note in dihedrals:
            atoms, degree = note
            rdMolTransforms.SetDihedralRad(mol.GetConformer(), *atoms, degree)
        if(full_block):
            return Chem.MolToXYZBlock(mol)
        return '\n'.join(Chem.MolToXYZBlock(mol).split('\n')[2:])
    except OSError:
        print("No such file!")
        return None

def to_degrees(dihedrals : list[dihedral]) -> list[dihedral]:
    """
        Convert rads to degrees in dihedrals
    """
    res = []
    for cur in dihedrals:
        a, d = cur
        res.append((a, d * 180 / math.pi))
    print(res)
    return res

def read_xyz(name : str) -> list[str]:
    """
        read coords from 'filename' and return that as a list of strings
    """
    xyz = []
    with open(name, 'r') as file:
        for line in file:
            xyz.append(line)
    return '\n'.join(xyz)

def generate_gjf(coords : str,
                 gjf_name : str,
                 num_of_procs : int,
                 method_of_calc : str,
                 charge : int,
                 multipl : int):
    """
        coords - xyz coords bloack with '\n' sep
        generate .gjf file for calculation energy of 'coords' by 'method_of_calc'
    """
    with open(gjf_name, 'w+') as file:
        file.write("%nprocs=" + str(num_of_procs) + "\n" + "\n")
        file.write("#P " + method_of_calc + "\n" + "\n")
        file.write(gjf_name + "\n" + "\n")
        file.write(str(charge) + " " + str(multipl) + "\n")
        file.write(coords)
        file.write("\n" + "\n")

def generate_default_gjf(coords : str, gjf_name : str):
    """
        generate .gjf with default properties
    """
    generate_gjf(coords, gjf_name, DEFAULT_NUM_OF_PROCS, DEFAULT_METHOD,\
                                         DEFAULT_CHARGE, DEFAULT_MULTIPL)

def generate_oinp(coords : str, 
                  dihedrals : list[dihedral],
                  gjf_name : str, 
                  num_of_procs : int, 
                  method_of_calc : str,
                  charge : int,
                  multipl : int):
    """
        generates orca .inp file
    """
    with open(gjf_name, 'w+') as file:
        file.write("!" + method_of_calc + " opt\n")
        file.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")
        file.write("%geom Constraints\n")
        dihedrals = to_degrees(dihedrals)
        for cur in dihedrals:
            a, d = cur
            file.write("{ D " + " ".join(map(str, a)) + " " + str(d) + " C }\n")
        file.write("end\nend\n")    
        file.write("* xyz " + str(charge) + " " + str(multipl) + "\n")
        file.write(coords)
        file.write("END\n")

def generate_default_oinp(coords : str, dihedrals : list[dihedral], oinp_name : str):
    generate_oinp(coords, dihedrals, oinp_name, DEFAULT_NUM_OF_PROCS, DEFAULT_ORCA_METHOD,\
                                                DEFAULT_CHARGE, DEFAULT_MULTIPL)
def start_calc(gjf_name : str):
    """
        Running calculation
    """	
    if not USE_ORCA:
        os.system(GAUSSIAN_EXEC_COMMAND + " " + gjf_name)
    else:
        sbatch_name = gjf_name.split('/')[-1][:-4] + ".sh"
        os.system("cp sbatch_temp " + sbatch_name)
        os.system("echo \"" + ORCA_EXEC_COMMAND + " " + gjf_name + " > " + gjf_name[:-4] + ".out\"" + " >> " + sbatch_name) 
        os.system("sbatch " + sbatch_name)    

def gjf_to_log_name(gjf_name : str) -> str:
    """
        generating name of log file from gjf file name
    """
    return gjf_name[:-4] + ".log"

def mol_to_gjf_name(mol_file_name : str) -> str:
    """
        generating name of gjf file from mol file name
    """
    return mol_file_name[:-4] + ".gjf"

def mol_to_inp_name(mol_file_name : str) -> str:
    """
        generating name of inp file from mol file name
    """
    return mol_file_name[:-4] + ".inp"

def inp_to_out_name(inp_file_name : str) -> str:
    """
        generating name of out file from inp file name
    """
    return inp_file_name[:-4] + ".out"

def wait_for_the_end_of_calc(log_name : str, timeout):
    """
        waiting fot the end of calculation in gaussian 
        by checking log file every 'timeout' ms
    """
    while True:
        try: 
            with open(log_name, 'r') as file:
                log_file = [line for line in file]               
                if(not USE_ORCA and ("Normal termination" in log_file[-1] or\
                   "File lengths" in log_file[-1])):
                    break
                if(USE_ORCA and ("ORCA TERMINATED NORMALLY" in log_file[-2] or\
                                 "ORCA finished by error" in log_file[-5])):
                    break
        except FileNotFoundError:
            pass
        except IndexError:
            pass
        finally:
            time.sleep(timeout / 1000)

def find_energy_in_log(log_name : str) -> float:
    """
        finds energy of structure in log file
    """
    with open(log_name, 'r') as file:
        if not USE_ORCA:
            en_line = [line for line in file if "SCF Done" in line][0]
            en = float(en_line.split()[4])
        else:
            en_line = [line for line in file if "FINAL SINGLE POINT ENERGY" in line][-1]
            en = float(en_line.split()[4])
        return en
    
def calc_energy(mol_file_name : str,
                dihedrals : list[dihedral] = []) -> float:
    """
        calculates energy of molecule from 'mol_file_name'
        with current properties and returns it as float
    """
    xyz_upd = change_dihedrals(mol_file_name, dihedrals)
    if(xyz_upd is None):
        return None
    if not USE_ORCA:
        gjf_name = mol_to_gjf_name(mol_file_name)
        generate_default_gjf(xyz_upd, gjf_name)
        start_calc(gjf_name)
        log_name = gjf_to_log_name(gjf_name)
        wait_for_the_end_of_calc(log_name, 250)
        res = find_energy_in_log(log_name)
        os.system("rm -r " + log_name)
    else:
        inp_name = mol_to_inp_name(mol_file_name)
        generate_default_oinp(xyz_upd, dihedrals, inp_name)
        start_calc(inp_name)
        out_name = inp_to_out_name(inp_name)
        wait_for_the_end_of_calc(out_name, 1000)
        res = find_energy_in_log(out_name)
        os.system("rm -r " + out_name)
    return res

mol_name = "tests/cur.mol"
#dh = [([0, 1, 2, 3], math.pi)]
#xyz = change_dihedrals(mol_name, dh)
#generate_default_oinp(xyz, dh, mol_name[:-4] + ".inp")
#start_calc(mol_name[:-4] + ".inp")
#wait_for_the_end_of_calc(mol_name[:-4] + ".out", 1000)
#print(find_energy_in_log(mol_name[:-4] + ".out"))
print(calc_energy("tests/cur.mol", [([0, 1, 2, 3], 2 * math.pi)]))
