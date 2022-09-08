import argparse
import os
cwd = os.getcwd()


class BaseOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # data directory
        self.parser.add_argument("-i", "--input", type=str, default=cwd, help="Path to input file with SMILES")
        self.parser.add_argument("-o", "--output_dir", type=str, default=cwd, help="Name of directory to save models")

        # ligand grid parameters
        self.parser.add_argument("--grid_resol", type=int, default=1.0, help="Grid resolution")
        self.parser.add_argument("--grid_dim", type=int, default=23.0, help="Grid dimension")
        self.parser.add_argument('--rotation', action='store_true', help="To not rotate data don't include --rotation in args")

        # training
        self.parser.add_argument("--epochs", type=int, default=7, help="number of epochs")
        self.parser.add_argument("--batch_size", type=int, default=100, help="input batch size")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
        self.parser.add_argument("--start_cap", type=int, default=2, help="iteration to start training caption")

        # evaluation
        self.parser.add_argument("--dir_ligshape", type=str, default=cwd, help="directory with ligand shapes")
        self.parser.add_argument("--num_ligs", type=int, default=100, help="number of ligand to be generated")

        # hyperparameters
        self.parser.add_argument("--lr_cap", type=float, default=0.001, help="initial learning rate for adam")
        self.parser.add_argument("--lr_VAE", type=float, default=0.0005, help="initial learning rate for adam")

        # GPU ids
        self.parser.add_argument("--GPUIDS", nargs="+", type=int, default=[0] ,help="GPU ids available, example 0 1 2")


    def create_parser(self):
        return vars(self.parser.parse_args())

args = BaseOptions().create_parser()

print(args)
