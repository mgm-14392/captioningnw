from io import StringIO
from rdkit import Chem
import h5py
import gzip
import sys

#in_path = sys.argv[1] # gz file
#namehdf5db = sys.argv[2] # name of database

def store_mol(smile, mol_block, h5file):
    data = [n.encode("ascii", "ignore") for n in mol_block]
    h5file.create_dataset(smile, data=data, compression="gzip")


def get_mol_data(m):
    mol_block = Chem.MolToMolBlock(m).splitlines(True)
    mol_id = m.GetProp('_Name')
    SMILES = m.GetProp('smiles')
    print(SMILES, mol_id)
    return SMILES, mol_block


def gen_hdf5db(in_path, namehdf5db, file_w_or_a):
    with h5py.File(namehdf5db, file_w_or_a, libver='latest') as h5file:
        with Chem.ForwardSDMolSupplier(gzip.open(in_path,'rb')) as suppl:
            for m in suppl:
                try:
                    SMILES, mol_block = get_mol_data(m)
                    store_mol(SMILES, mol_block, h5file)
                except:
                    pass


gen_hdf5db("selected_aa.sdf.gz","capNWdatav3.hdf5","w")
gen_hdf5db("selected_ab.sdf.gz","capNWdatav3.hdf5","a")
gen_hdf5db("selected_ac.sdf.gz","capNWdatav3.hdf5","a")
