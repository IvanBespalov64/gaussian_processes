GAUSSIAN_EXEC_COMMAND = ""
DEFAULT_METHOD = "RHF/STO-3G"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPL = 1

def read_xyz(name : str) -> list[str]:
    """
    read coords from 'filename' and return that as a list of strings
    """
    xyz = []
    with open(name) as file:
        for line in file:
            xyz.append(line)
    return xyz

def generate_gjf(coords : list[str],
                 gjf_name : str,
                 method_of_calc : str,
                 charge : int,
                 multipl : int):
    """
    generate .gjf file for calculation energy of 'coords' by 'method_of_calc'
    """
    with open(gjf_name, 'w+') as file:
        file.write("#P " + method_of_calc + "\n" + "\n")
        file.write(gjf_name + "\n" + "\n")
        file.write(str(charge) + " " + str(multipl) + "\n")
        for line in coords:
            file.write(line)
        file.write("\n" + "\n")

def generate_dafault_gjf(coords : list[str], gjf_name : str):
    """
    generate gjf with default properties
    """
    generate_gjf(coords, gjf_name, DEFAULT_METHOD, DEFAULT_CHARGE, DEFAULT_MULTIPL)

filename = input("Enter xyz file name: ")
xyz = []

try:
    xyz = read_xyz(filename)
except FileNotFoundError:
    print("No such file!")
    exit()

generate_dafault_gjf(xyz, "temp.gjf")
