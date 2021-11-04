"""Create a list of sequence padded 3D matrices containing neighbor features."""

# modules
import torch
from torch.nn.utils.rnn import pad_sequence
from cgcnn.data import CIFData
from os.path import join


def nbr_feat(data=join("data", "sample-regression")):
    """Create CGCNN neighbor feature matrix.

    Parameters
    ----------
    data : str or DataFrame
        Path to folder containing CIFs or DataFrame with pymatgen Structures.

    Returns
    -------
    total_nbr_fea_master : 4D Array
        CGCNN neighbor feature matrix.

    TODO: describe dimensions of the 4D matrix
    """
    # import and process data
    if type(data) is str:
        dataset = CIFData(data)
    else:
        1 + 1
        # TODO: implement use of DataFrames

    # extract neighbor features
    for i, data in enumerate(dataset):
        # unpack
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = data

        # taken from model.py --> class ConvLayer --> forward
        N, M = nbr_fea_idx.shape
        if i == 1:
            atom_fea_len = atom_fea.shape[1]

        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_fea.unsqueeze(1).expand(N, M, atom_fea_len), atom_nbr_fea, nbr_fea],
            dim=2,
        )

        # construct list
        if i == 0:
            # initialize
            total_nbr_fea_list = [total_nbr_fea]
        else:
            # append
            total_nbr_fea_list.append(total_nbr_fea)

    # pad with zeros and concatenate
    total_nbr_fea_master = pad_sequence(total_nbr_fea_list, batch_first=True).numpy()

    return total_nbr_fea_master


"""code graveyard"""
"""
# print((cif_id, nbr_fea.shape, atom_fea.shape))

        #nbr_fea_idx_list = [nbr_fea_idx]
        #nbr_fea_idx_list.append(nbr_fea_idx)
        
#nbr_fea_idx_master = pad_sequence(total_nbr_fea_idx_list, batch_first=True).numpy()
"""
