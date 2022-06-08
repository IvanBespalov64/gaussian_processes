from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import gpflow

import default_vals
import calc

from mean_cos import BaseCos

class CoefCalculator:
    """
        This class performs splitting given molecule on
        small parts with only one interesting rotable dihedral.
        Scanning energies of torsion rotation with given method.
        Calculating coefs for mean function for GPRegressor
    """

    def __init__(self,
                 mol : Chem.rdchem.Mol,
                 dir_for_inps : str="",
                 skip_triple_equal_terminal_atoms=True,
                 num_of_procs : int = default_vals.DEFAULT_NUM_OF_PROCS,
                 method_of_calc : str = default_vals.DEFAULT_ORCA_METHOD,
                 charge : int = default_vals.DEFAULT_CHARGE,
                 multipl : int = default_vals.DEFAULT_MULTIPL,
                 degrees : np.ndarray = np.linspace(0, 2 * np.pi, 37).reshape(37, 1)):
        """
            mol - rdkit molecule
            dir_for_inps - path to directory, where scan .inp files will generates
            skip_triple_equal_terminal_atoms - skip diherdrals,
                where one of atoms is RX3, where X - terminal atom
            num_of_procs - num of procs to calculate
            method_of_calc - method in orca format
            charge - charge of molecule
            multipl - multiplicity
            degrees - degree grid to scan
        """

        self.mol = mol
        self.dir_for_inps = dir_for_inps if dir_for_inps[-1] == "/" else dir_for_inps + "/"
        self.skip_triple_equal_terminal_atoms = skip_triple_equal_terminal_atoms
        self.num_of_procs = num_of_procs
        self.method_of_calc = method_of_calc
        self.charge = charge
        self.multipl = multipl
        self.degrees = degrees

        # Key is SMILES, val is idx
        self.unique_frags = {}
        # Key is atom idxs, val is idx
        self.frags = {}

    def is_terminal(self,
                    atom : Chem.rdchem.Atom) -> bool:
        """
            Returns True if atom is terminal(Hs not counted)
        """
        return len(atom.GetNeighbors()) == 1

    def get_second_atom_in_bond(self,
                                bond : Chem.rdchem.Bond,
                                atom : Chem.rdchem.Atom) -> Chem.rdchem.Atom:
        """
            retruns another atom from this bond
        """
        return bond.GetEndAtom() if bond.GetBeginAtom() == atom else bond.GetBeginAtom()

    def is_triple_eq_neighbors(self,
                               atom : Chem.rdchem.Atom) -> bool:
        """
        check if current atom has three equal neighbors

        """

        in_bond = None

        for bond in atom.GetBonds():
            if not self.is_terminal(self.get_second_atom_in_bond(bond, atom)):
                in_bond = bond
                break

        if in_bond is None:
            return False

        neighbor_atoms = [cur.GetSymbol() for cur in atom.GetNeighbors()]
        neighbor_atoms.remove(self.get_second_atom_in_bond(in_bond, atom).GetSymbol())

        terminal_neighbors = False

        # 3 terminal neighbors
        if sum([self.is_terminal(cur) for cur in atom.GetNeighbors()]) == 3:
            terminal_neighbors = True

        if(terminal_neighbors and len(neighbor_atoms) == 3 and len(set(neighbor_atoms)) == 1):
            return True

        return False

    def is_interesting(self,
                       bond : Chem.rdchem.Atom) -> bool:
        """
            Returns True if we should scan this bond
            if skip_triple_equale_terminal_atoms == True - dihedral
            angels, where on one atom there are three equal terminal atoms,
            are not interesting
        """

        #If one of atoms is terminal
        if len([cur for cur in bond.GetBeginAtom().GetBonds()]) < 2 or\
           len([cur for cur in bond.GetEndAtom().GetBonds()]) < 2 :
            return False

        # If bond isn't single
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False

        if not self.skip_triple_equal_terminal_atoms:
            return True

        # If one of atoms has three equal terminal atom neighbors
        for t_atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if self.is_triple_eq_neighbors(t_atom):
                return False

        return True

    def get_unique_mols(self,
                        lst : list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
        """
            get unique molecules from list
        """

        return list(map(Chem.MolFromSmiles, set(list(map(Chem.MolToSmiles, lst)))))

    def generate_3d_coords(self,
                           lst : list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
        """
            returns list with same molecules but with
            Hs and generated coords by ETKDG
        """

        result = []

        for mol in lst:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            result.append(mol)

        return result

    def get_idxs_to_rotate(self,
                           mol : Chem.rdchem.Mol) -> list[int]:
        """
            Returns idxs of dihedral angel in correct order
        """

        for bond in mol.GetBonds():

            if not self.is_interesting(bond):
                continue

            return ([cur.GetIdx() for cur in bond.GetBeginAtom().GetNeighbors() if cur.GetIdx() != bond.GetEndAtomIdx()][0],
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    [cur.GetIdx() for cur in bond.GetEndAtom().GetNeighbors() if cur.GetIdx() != bond.GetBeginAtomIdx()][0])

    def get_interesting_frags(self) -> list[Chem.rdchem.Mol]:
        """
            returns a list of simple molecules with one
            rotable interesting torsion angle
            if skip_triple_equale_terminal_atoms == True - dihedral
            angels, where on one atom there are three equal terminal atoms,
            are not interesting
        """

        rotable_frags = []

        count = 0

        for bond in self.mol.GetBonds():
            if not self.is_interesting(bond):
                continue

            atoms_to_use = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

            for atom in [*bond.GetBeginAtom().GetNeighbors(),\
                         *bond.GetEndAtom().GetNeighbors()]:
                atoms_to_use.add(atom.GetIdx())

            if self.skip_triple_equal_terminal_atoms and\
               self.is_triple_eq_neighbors(atom):
                atoms_to_use.update([cur.GetIdx() for cur in atom.GetNeighbors()])

            rotable_frags.append(Chem.MolFromSmiles(
                Chem.rdmolfiles.MolFragmentToSmiles(self.mol, atomsToUse = list(atoms_to_use))))

            query_result = self.mol.GetSubstructMatches(Chem.MolFromSmiles(
                                                        Chem.rdmolfiles.MolFragmentToSmiles(rotable_frags[-1],
                                                                                            atomsToUse = self.get_idxs_to_rotate(rotable_frags[-1]))))
            old_idxs = ()

            for res in query_result:
                corr_idxs = True
                for cur in res:
                    corr_idxs = corr_idxs and cur in atoms_to_use
                if corr_idxs:
                    old_idxs = res
                    break

            frag_smiles = Chem.MolToSmiles(rotable_frags[-1])

            if frag_smiles in self.unique_frags:
                self.frags[old_idxs] = self.unique_frags[frag_smiles]
            else:
                self.unique_frags[frag_smiles] = count
                self.frags[old_idxs] = count
                count += 1

        return self.generate_3d_coords(self.get_unique_mols(rotable_frags))

    def get_list_of_xyz(self,
                        lst : list[Chem.rdchem.Mol]) -> list[str]:
        """
            returns list of xyz-blocks of given molecules
        """

        return list(map(Chem.MolToXYZBlock, lst))

    def generate_scan_inp(self,
                          xyz : str,
                          idxs_to_rotate : list[int],
                          filename : str):
        """
            Generates .inp file with "filename" for scan
            of mol, described by "xyz" xyz block, in orca
            Note that we rotate 0-1-2-3 angle, I think,
            that it should work always
        """
        with open(filename, 'w+') as file:
            file.write("!" + self.method_of_calc + " opt\n")
            file.write("%pal\nnprocs " + str(self.num_of_procs) + "\nend\n")
            file.write("%geom Scan\n")
            file.write("D " + " ".join(list(map(str, idxs_to_rotate))) + " = 0.0, 360.0, 37\n")
            file.write("end\nend\n")
            file.write("* xyz " + str(self.charge) + " " + str(self.multipl) + "\n")
            file.write(xyz)
            file.write("END\n")

    def get_coords_from_xyz_block(self,
                                  xyz : str) -> str:
        """
            returns xyz-coords from xyz block
            erase first info lines
        """

        return "\n".join(xyz.split("\n")[2:])

    def generate_scan_inps_from_mol(self) -> list[str]:
        """
            Generates inp files for scanning of all interesting
            unique fragments from molecule.
            Returns list of .inp filenames
            dir_for_inps - path of directory including folders separator
        """

        inp_names = []

        angle_number = 0

        for sub_mol in self.get_interesting_frags():
            xyz = Chem.MolToXYZBlock(sub_mol)
            idxs_to_rotate = self.get_idxs_to_rotate(sub_mol)
            filename = self.dir_for_inps + "scan_" + str(angle_number) + ".inp"
            self.generate_scan_inp(self.get_coords_from_xyz_block(xyz), idxs_to_rotate, filename)
            inp_names.append(filename)
            angle_number += 1

        return inp_names

    def get_energies_from_scans(self,
                                lst : list[str]) -> np.ndarray:
        """
            lst - list of input file paths,
            return list of lists of energies in
            [0.0, 360.0] with step = 10 degrees
        """

        out_names = list(map(lambda s : s[:-3] + "out", lst))
        for out_name in out_names:
            calc.wait_for_the_end_of_calc(out_name, 1000)

        res_file_names = list(map(lambda s : s[:-3] + "relaxscanact.dat", lst))

        result = []

        for res_file in res_file_names:
            cur_res = []
            with open(res_file, "r") as file:
                for line in file:
                    cur_res.append(float(line[:-1].split()[1]))
            result.append(np.array(cur_res))

        return np.array(result)

    def get_scans_of_dihedrals(self) -> np.ndarray:
        """
            Returns list of energie dependecies
        """

        inp_files = self.generate_scan_inps_from_mol()
        #for cur in inp_files:
        #    calc.start_calc(cur)
        return self.get_energies_from_scans(inp_files)

    def get_coefs(self,
                  energies : np.ndarray) -> list[float]:
        """
            Calculates coefs of mean function for
            given dependency of energy from degree.
            By default degrees is [0., 2 * np.pi] with step 10
        """

        #Normalize energies
        energies = (energies - np.min(energies)) * 627.509474063

        model = gpflow.models.GPR((self.degrees.astype('double'),
                                   energies.reshape(self.degrees.shape[0], 1).astype('double')),
                                   gpflow.kernels.Periodic(gpflow.kernels.Matern12(), 2 * np.pi),
                                   BaseCos(1))

        gpflow.optimizers.Scipy().minimize(
            model.training_loss,
            variables=model.trainable_variables)

        return [param.numpy().flatten()[0] for param in model.mean_function.parameters]

    def calc(self) -> list[list[float]]:
        """
            Calculate coefs for mean function
        """

        res = []

        for energies in self.get_scans_of_dihedrals():
            res.append(self.get_coefs(energies))

        print(self.unique_frags)
        print(self.frags)        

        return res

    def coef_matrix(self) -> list[tuple[tuple, list[float]]]:
        """
            Get matrix of coefficients for mean function 
            for all dihedral angels 
        """

        unique_coefs = self.calc()
        result = []        

        for idxs in self.frags:
            result.append((list(idxs), unique_coefs[self.frags[idxs]]))

        return result

#print(CoefCalculator(Chem.MolFromMolFile("tests/cur.mol"),
#                                         "test_scans/").coef_matrix())
#print(CoefCalculator(Chem.MolFromSmiles("O=CCCCCCC(N)(C(F)(F)(F))"), "test_tscans/").coef_matrix())
