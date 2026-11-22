from rdkit import Chem
from func_group_set import load_fgc, decompose_mol, decompose_mol1


group_set = load_fgc("../data/config1.fgc")

import csv

with open("../data/pfas.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    column = [row["SMILES"] for row in reader]


def diagnose_decomposition(mol, decomposition):
    assigned = set()

    # print("Functional group assignments:")

    for count, group in decomposition:
        matches = group.get_matches(mol)

        # print(f"\n{group}")
        # print(f"Reported count: {count}")

        for match in matches:
            if assigned.intersection(match):
                continue

            assigned.update(match)
            # print(f"  Assigned atoms: {match}")

            if len(assigned) >= mol.GetNumAtoms():
                break

    unassigned = set(range(mol.GetNumAtoms())) - assigned

    # print("\nUnassigned atoms:")
    # print(sorted(unassigned))
    #
    # print(
    #     f"Coverage: {len(assigned)}/{mol.GetNumAtoms()} "
    #     f"({100 * len(assigned) / mol.GetNumAtoms():.1f}%)"
    # )

    return len(assigned)/mol.GetNumAtoms()

total = 0
coveraged = []

for i in range(0, 21000):
    smiles = column[i]
    mol = Chem.MolFromSmiles(smiles)
    if not mol: continue
    decomp = decompose_mol1(mol, group_set)
    coveraged.append(diagnose_decomposition(mol, decomp))

import numpy as np

coverage = np.array(coveraged) * 100.0

print(f"Mean:   {coverage.mean():.2f}%")
print(f"Median: {np.median(coverage):.2f}%")
print(f"Std:    {coverage.std():.2f}%")


# Mean:   75.46%
# Median: 84.00%
# Std:    24.14%




#####

# Mean:   0.76%
# Median: 0.85%
# Std:    0.23%

#
# Mean:   79.99%
# Median: 90.00%
# Std:    23.44%

##
# Mean:   80.06%
# Median: 90.00%
# Std:    23.47%

###
# Mean:   81.28%
# Median: 92.31%
# Std:    23.36%

