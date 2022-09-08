import molgrid
from molgrid import Coords2GridFunction
from molgrid import GridMaker
import baseoptions
import numpy as np
import torch
import pybel
import openbabel
from rdkit import Chem
import time
import os
import h5py
from scipy.spatial.transform import Rotation
import random

#args = baseoptions.BaseOptions().create_parser()

def makecanonical(text):
    """Make smiles canonical"""

    mol = Chem.MolFromSmiles(text)
    if not mol:
        raise ValueError("Failed to parse molecule '{}'".format(mol))
    sstring = Chem.MolToSmiles(mol, canonical= False, isomericSmiles = False)
    return sstring


def smiles_to_np(text):
    """ convert SMILES into tensor """

    vocab_list_i = ["pad", "start", "end",
                    "C", "c", "N", "n", "S", "s", "P", "O", "o",
                    "B", "F", "I",
                    "X", "Y", "Z",
                    "1", "2", "3", "4", "5", "6",
                    "#", "=", "-", "(", ")"
                    ]
    vocab_i2c_v1_i = {i: x for i, x in enumerate(vocab_list_i)}
    vocab_c2i_v1_i = {vocab_i2c_v1_i[i]: i for i in vocab_i2c_v1_i}

    strings = np.zeros((72),dtype='uint8')

    sstring = makecanonical(text)

    sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z").replace("[PH]","P").replace("[SH]","S").replace("[S]", "S")

    try:
        vals = [1] + [vocab_c2i_v1_i[xchar] for xchar in sstring] + [2]
    except KeyError:
        raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                          .format(", ".join([x for x in sstring if x not in vocab_c2i_v1_i]),
                                  sstring)))

    end_token = vals.index(2)

    strings[0 : len(vals)] =vals

    stensor = torch.from_numpy(strings)
    stensor = stensor.type(torch.LongTensor)
    return stensor, end_token + 1


def get_mol(smile, h5file):
    dataset = h5file[smile]
    endstring = ''
    for line in dataset:
        #print(line.rstrip().decode())
        endstring += line.decode()
    return endstring


def coords2grid(text, hdf5_file, resolution = 23, dimension = 1.0, rotation = True):
    smiles = text.strip()
    g_mol = get_mol(smiles, hdf5_file)
    mol = pybel.readstring('sdf', g_mol)

    # ROTATION
    c = molgrid.CoordinateSet(mol)

    if rotation:
        t = molgrid.Transform(molgrid.Quaternion(random.uniform(0,10),random.uniform(0,10),random.uniform(0,10),random.uniform(0,10)), (0,0,0), (0,0,0))
        t.forward(c, c)

    gmaker = molgrid.GridMaker(resolution = resolution, dimension = dimension, binary=False,
                               radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

    dims = gmaker.grid_dimensions(molgrid.defaultGninaLigandTyper.num_types())
    gridtensor = torch.zeros(dims,dtype=torch.float32)
    gmaker.forward(c.center(),c,gridtensor)

    #NO ROTATION
    #crds = molgrid.CoordinateSet(mol.OBMol, molgrid.defaultGninaLigandTyper)
    #crds.make_vector_types()
    #center = tuple(crds.center())
    #coordinates = torch.tensor(crds.coords.tonumpy())
    #radii = torch.tensor(crds.radii.tonumpy())
    #types = torch.tensor(crds.type_vector.tonumpy())

    #gmaker = molgrid.GridMaker(resolution = resolution, dimension = dimension, binary=False,
    #                       radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

    #grid = Coords2GridFunction.apply(gmaker, center, coordinates, types, radii)

    #torch.save(grid, "NOTrot_check.pt")
    return gridtensor


if __name__ == '__main__':
    args = baseoptions.BaseOptions().create_parser()
    coords2grid(text, hdf5_file, resolution = args['grid_resol'], dimension = args['grid_dim'], rotation = args['rotation'])

    #smi = 'C=CCN1C(=O)c2ccc(C(=O)OCC(=O)N3CCN(S(=O)(=O)c4ccccc4)CC3)cc2C1=O'
    #_h5_gen = h5py.File('/c7/scratch2/Mariana/cnns/datacaptionnetwork/MolPort_ligs/Prep/capNWdatav3.hdf5', 'r')
    #grid = coords2grid(smi, _h5_gen, resolution =1.0, dimension = 23, rotation=True)
    #torch.save(grid, "NOrot.pt")
