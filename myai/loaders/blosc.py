import torch
import bloscpack as bp

def bloscread(filename):
    return bp.unpack_ndarray_from_file(filename)

def bloscreadtensor(filename):
    return torch.from_numpy(bloscread(filename))

def bloscwrite(filename, array, clevel = 2, shuffle=True):
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    blosc_args = bp.BloscArgs()
    blosc_args.clevel = clevel
    blosc_args.shuffle = shuffle
    bp.pack_ndarray_to_file(ndarray = array, filename = filename,  blosc_args=blosc_args)