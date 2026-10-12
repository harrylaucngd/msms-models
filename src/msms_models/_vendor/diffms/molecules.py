from rdkit import Chem
import rdkit.Chem

class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol
