from io import StringIO
from rdkit import Chem
import h5py
import gzip
import sys

in_path = sys.argv[1] # gz file
nameofile = sys.argv[2] # name of database


def convert_to_molblock(mol):
    sio = StringIO()
    with Chem.SDWriter(sio) as w:
        w.write(mol)
    contents = sio.getvalue()
    sio.close()
    return contents


def store_mol(smile, mol_block, h5file):
    #print(mol_block)
    data = [n.encode("ascii", "ignore") for n in mol_block.splitlines(True)]
    h5file.create_dataset(smile, data=data, compression="gzip")


with h5py.File(nameofile, 'w') as h5file:
    with Chem.ForwardSDMolSupplier(gzip.open(in_path,'rb'), sanitize=False) as suppl:
        for m in suppl:
            try:
                mol_block = convert_to_molblock(m)
                mol_id = m.GetProp('_Name')
                SMILES = m.GetProp('smiles')
                print(SMILES, mol_id)
                #store_mol(SMILES, mol_block, h5file)
            except:
                print('molblock extraction failed for mol in sdf')

