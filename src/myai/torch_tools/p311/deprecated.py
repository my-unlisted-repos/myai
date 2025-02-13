import torch

def crop_around(tensor:torch.Tensor, coord, size) -> torch.Tensor:
    """Returns a tensor of `size` size around `coord`"""
    if len(coord) == 3:
        x, y, z = coord
        x, y, z = int(x), int(y), int(z)
        sx, sy, sz = size
        sx, sy, sz = int(sx//2), int(sy//2), int(sz//2)
        if tensor.ndim == 3: shape = tensor.size()
        elif tensor.ndim == 4: shape = tensor.shape[1:]
        else: raise NotImplementedError

        if x-sx < 0: x = x - (x-sx)
        if y-sy < 0: y = y - (y-sy)
        if z-sz < 0: z = z - (z-sz)
        if x+sx+1 > shape[0]: x = x - (x+sx+1 - shape[0])
        if y+sy+1 > shape[1]: y = y - (y+sy+1 - shape[1])
        if z+sz+1 > shape[2]: z = z - (z+sz+1 - shape[2])
        if tensor.ndim == 3: return tensor[int(x-sx):int(x+sx), int(y-sy):int(y+sy), int(z-sz):int(z+sz)]
        elif tensor.ndim == 4:
            return tensor[:, int(x-sx):int(x+sx), int(y-sy):int(y+sy), int(z-sz):int(z+sz)]
        else: raise NotImplementedError

    elif len(coord) == 2:
        x, y = coord
        sx, sy = size
        sx, sy = int(sx/2), int(sy/2)
        if tensor.ndim == 2: shape = tensor.size()
        elif tensor.ndim == 3: shape = tensor.shape[1:]
        elif tensor.ndim == 4: shape = tensor.shape[2:]
        else: raise NotImplementedError

        if x-sx < 0: x = x - (x-sx)
        if y-sy < 0: y = y - (y-sy)
        if x+sx+1 > shape[0]: x = x - (x+sx+1 - shape[0])
        if y+sy+1 > shape[1]: y = y - (y+sy+1 - shape[1])
        if tensor.ndim == 2: return tensor[int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        elif tensor.ndim == 3:
            return tensor[:, int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        elif tensor.ndim == 4:
            return tensor[:,:, int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        else: raise NotImplementedError
    else: raise NotImplementedError
