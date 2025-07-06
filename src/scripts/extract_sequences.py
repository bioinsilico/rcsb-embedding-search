
import argparse
import os
import biotite.structure.io.pdb as pdb
from biotite.structure import to_sequence, filter_polymer, filter_amino_acids


def extract_sequences(input_dir, class_file, output_path=None):

    scop_map = {}
    print(f"Loading SCOPe map {class_file}")
    for row in open(class_file):
        r = row.strip().split("\t")
        scop_map[r[0]] = r[1]

    records = []
    for pdb_file in os.listdir(input_dir):
        pdb_struct = pdb.PDBFile.read(f"{input_dir}/{pdb_file}")
        atom_array = pdb_struct.get_structure()
        atom_array = atom_array[filter_polymer(atom_array)]
        atom_array = atom_array[filter_amino_acids(atom_array)]
        sequences, chain_starts = to_sequence(atom_array)
        for seq, start_idx in zip(sequences, chain_starts):
            chain_id = atom_array.chain_id[start_idx]
            header = f"{pdb_file}|{chain_id}|{scop_map[pdb_file]}"
            records.append((header, str(seq)))
    if output_path:
        with open(output_path, "w") as f:
            for header, seq in records:
                f.write(f">{header}\n{seq}\n")
    else:
        for header, seq in records:
            print(f">{header}\n{seq}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract protein sequences from PDB files in a directory using Biotite."
    )
    parser.add_argument(
        "--pdb-path",
        required=True,
        help="Path to the directory containing .pdb files"
    )
    parser.add_argument(
        "--scop-class-file",
        required=True,
        help="File to the SCOPe classification of domains"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output FASTA file (default: stdout)",
        default=None
    )
    args = parser.parse_args()
    extract_sequences(args.pdb_path, args.scop_class_file, args.output)

if __name__ == "__main__":
    main()
