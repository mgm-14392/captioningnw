import molgrid
from molgrid import Coords2GridFunction
from io import StringIO
import pybel
from rdkit import Chem
import torch
import h5py
import pandas as pd
import numpy as np
import sys

h5db = sys.argv[1]
SMILES = sys.argv[2]

def get_mol(smile, h5file):
    dataset = h5file[smile]
    endstring = ''
    for line in dataset:
        #print(line.rstrip().decode())
        endstring += line.decode()
    return endstring


with h5py.File(h5db, 'r') as f:

    #get_keys = list(f.keys())
    #print(get_keys)
    #print(len(get_keys))
    g_mol = get_mol(SMILES, f)
    print(g_mol)

    mol = pybel.readstring('sdf',g_mol)
    print(mol)
    print(mol.molwt)
    print(len(mol.atoms))

    crds = molgrid.CoordinateSet(mol.OBMol, molgrid.defaultGninaLigandTyper)
    crds.make_vector_types()
    center = tuple(crds.center())
    coordinates = torch.tensor(crds.coords.tonumpy())
    radii = torch.tensor(crds.radii.tonumpy())
    types = torch.tensor(crds.type_vector.tonumpy())

    # initialize gridMaker
    gmaker = molgrid.GridMaker(resolution = 1.0, dimension = 23, binary=False,
                            radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

    grid = Coords2GridFunction.apply(gmaker, center, coordinates, types, radii)
    print(grid.shape)
    torch.save(grid, 'mol.pt')
                                                                                               50,5          Bot
