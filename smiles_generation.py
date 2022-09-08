# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# Captioning network for BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# Modified from Ligdream (https://github.com/compsciencelab/ligdream)
# Modification of the original code to use libmolgrid for input preparation 8/04/22

from networks import EncoderCNN_v3, DecoderRNN, VAE
from generators_3 import makecanonical, get_mol, coords2grid
from decoding import decode_smiles
from torch.autograd import Variable
import molgrid
from molgrid import Coords2GridFunction
import torch
import pybel
from rdkit import Chem
from rdkit import RDLogger
import sys
import h5py
import Levenshtein as lev
from os import listdir
from os.path import isfile, join, isdir
import sys
import baseoptions

def normVoid_tensor(input_tensor):
    # normalize?? this does not add to 1
    input_tensor /= input_tensor.sum(axis = 1).max()
    # add void dimension
    input_tensor = torch.cat((input_tensor, (1 - input_tensor.sum(dim = 1).unsqueeze(1))), axis=1)
    # clamp values
    input_tensor = torch.clamp(input_tensor, min=0, max=1)
    # normalize again adds to one
    input_tensor /= input_tensor.sum(axis = 1).unsqueeze(1)
    return input_tensor


def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """

    RDLogger.DisableLog('rdApp.*')
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids

    #return [ Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in set(xresults)]  # Check for duplicates and filter out invalids
    return xresults

def get_smi_3D_voxels(smiles, filename_hdf5=None, evaluate=False):
    # SMILES to 3D with openbabel
    if evaluate:
        print('Your input SMILES: ', smiles)

        smiles = makecanonical(smiles)
        mol = pybel.readstring('smi', smiles)
        mol.addh()

        print('Generating 3D structure from SMILES')
        ff = pybel._forcefields["mmff94"]
        success = ff.Setup(mol.OBMol)
        if not success:
            ff = pybel._forcefields["uff"]
            success = ff.Setup(mol.OBMol)
            if not success:
                sys.exit("Cannot set up forcefield")

        ff.ConjugateGradients(100, 1.0e-3)  # optimize geometry
        ff.WeightedRotorSearch(100, 25)  # generate conformers
        ff.ConjugateGradients(250, 1.0e-4)
        ff.GetCoordinates(mol.OBMol)

        # get 3D coords
        #crds = molgrid.CoordinateSet(mol.OBMol, molgrid.defaultGninaLigandTyper)
        crds = molgrid.CoordinateSet(mol.OBMol)

    else:
        hdf5_file = h5py.File(filename_hdf5, 'r')
        g_mol = get_mol(smiles, hdf5_file)
        mol = pybel.readstring('sdf', g_mol)
        crds = molgrid.CoordinateSet(mol)


    gmaker = molgrid.GridMaker(resolution=1, dimension=23, binary=False,
                               radius_type_indexed=False, radius_scale=1.0,
                               gaussian_radius_multiple=1.0)

    dims = gmaker.grid_dimensions(molgrid.defaultGninaLigandTyper.num_types())
    gridtensor = torch.zeros(dims, dtype=torch.float32)
    gmaker.forward(crds.center(), crds, gridtensor)

    #crds.make_vector_types()
    #center = tuple(crds.center())
    #coordinates = torch.tensor(crds.coords.tonumpy())
    #radii = torch.tensor(crds.radii.tonumpy())
    #types = torch.tensor(crds.type_vector.tonumpy())

    # initialize gridMaker
    #gmaker = molgrid.GridMaker(resolution=1, dimension=23, binary=False,
    #                           radius_type_indexed=False, radius_scale=1.0,
    #                           gaussian_radius_multiple=1.0)

    #grid = Coords2GridFunction.apply(gmaker, center, coordinates, types, radii)
    print('Created ligand shape')
    return gridtensor


class CompoundGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False
        self.encoder = EncoderCNN_v3(15)
        self.decoder = DecoderRNN(512, 1024, 29, 1)
        self.vae_model = VAE()

        self.vae_model.eval()
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.vae_model.cuda()
            self.use_cuda = True


    def load_weight(self, vae_weights, cap_checkpoint):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.vae_model.load_state_dict(vae_weights['model_state_dict'])
        self.vae_model.eval()
        self.encoder.load_state_dict(cap_checkpoint['encoder_state_dict'])
        self.encoder.eval()
        self.decoder.load_state_dict(cap_checkpoint['decoder_state_dict'])
        self.decoder.eval()


    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions = torch.stack(captions, 1)
        if self.use_cuda:
            captions = captions.cpu().data.numpy()
        else:
            captions = captions.data.numpy()
        return decode_smiles(captions)


    def generate_molecules(self, shape, n_attemps=30, probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_valid: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """
        #print('Creating %d SMILES for the ligand shape' % n_attemps)
        shape_input = shape
        if self.use_cuda:
            shape_input = shape_input.cuda()

        shape_input = shape_input.repeat(n_attemps, 1, 1, 1, 1)

        shape_input = Variable(shape_input)

        recoded_shapes, _, _ = self.vae_model(shape_input)
        smiles = self.caption_shape(recoded_shapes, probab=probab)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)
        #return [Chem.MolFromSmiles(x) for x in smiles]
        return smiles

if __name__ == '__main__':
    args = baseoptions.BaseOptions().create_parser()

    filename_hdf5 = ''
    #smiles = 'C=CCN1C(=O)c2ccc(C(=O)OCC(=O)N3CCN(S(=O)(=O)c4ccccc4)CC3)cc2C1=O'
    #smiles = 'Cc1c(NC(=O)c2cc(F)cc(F)c2)nnn1Cc1ccccc1F'
    #smiles = 'c1ccccc1'
    #smiles = 'Cc1c(C(=O)N2c3ccccc3NC(=O)C2CC(=O)Nc2ccc(Cl)c(Cl)c2)oc2ccccc12'
    vae_weights = 'weights/VAE_5-35000.pth'
    cap_weights = 'weights/capNW_184-1152000.pth'
    vae_checkpoint = torch.load(vae_weights)
    cap_checkpoint = torch.load(cap_weights)

    comp_gen = CompoundGenerator()
    comp_gen.load_weight(vae_checkpoint, cap_checkpoint)

    #shape = get_smi_3D_voxels(smiles, None, True)
    #shapev = normVoid_tensor(shape.unsqueeze(0))

    #mypath = '/c7/home/margonza/Documents/cnns/scripts/scripts/shapenetwork/train/eval/a5a4/frames_shapes'
    mypath = args['dir_ligshape']
    desired_number_gencomps = args['num_ligs']
    #onlyfiles_all = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    #onlyfiles = [_file for _file in onlyfiles_all if _file.endswith('.pt') ]

    if isdir(mypath):
        onlyfiles_all = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        onlyfiles = [_file for _file in onlyfiles_all if _file.endswith('.pt') ]
    elif isfile(mypath):
        onlyfiles =[mypath]

    print(onlyfiles)
    for _file in onlyfiles:

        shapev = torch.load(_file)
        file_name = _file.rpartition("/")[-1].rpartition(".")[0]
        print(file_name)

        with open('%s.smi'%file_name, 'w') as the_file:

            #print('not prob')
            #for i in range(0,1000):
            #   generated_smiles = comp_gen.generate_molecules(shapev, n_attemps=40, filter_unique_valid=True)
            #   for smi in generated_smiles:
            #      print(smi)

            #lev = [lev.distance(smiles,i) for i in generated_smiles]
            #print('Levenshtein distance: ', lev)
            #print(len(generated_smiles)/40)

            #print('prob')
            smiles_list = []
            num_gen_correct_smiles = 0
            while num_gen_correct_smiles < desired_number_gencomps:
                prob_generated_smiles = comp_gen.generate_molecules(shapev, n_attemps=20, probab=True, filter_unique_valid=True)
                for smi in prob_generated_smiles:
                    smiles_list.append(smi)
                    the_file.write('%s\n' % smi)
                    num_gen_correct_smiles = len(smiles_list)

            #print('Generated SMILES: ', prob_generated_smiles)
            #print(len(generated_smiles)/40)
            #p_lev = [lev.distance(smiles,i) for i in prob_generated_smiles]
            #print('Levenshtein distance: ', p_lev)

