
from Bio.PDB import PDBParser
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa


def get_coords_from_pdb_file(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    return __get_coords_from_pdb_file(structure[0])


def __get_coords_from_pdb_file(structure):
    chains = [s.id for s in structure.get_chains()]
    coords = []
    for ch in chains:
        ca_atoms = [atom for atom in structure.get_atoms() if
                    atom.get_name() == "CA" and is_aa(atom.parent.resname) and atom.parent.parent.id == ch]
        if len(ca_atoms) < 10:
            continue
        coords.append({
            'cas': [atom.get_coord().tolist() for atom in ca_atoms],
            'seq':  [protein_letters_3to1_extended[c.parent.resname] for c in ca_atoms],
            'labels': [atom.full_id[3][1] for atom in ca_atoms]
        })
    return {
        'cas': [ch['cas'] for ch in coords],
        'seq': [ch['seq'] for ch in coords],
        'labels': [ch['labels'] for ch in coords]
    }



