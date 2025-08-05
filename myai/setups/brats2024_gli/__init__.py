import os
from ...python_tools import find_file_containing

BRATS2024_GLI_ROOT = r'E:\dataset\glio\BraTS-GLI v2\train'
def get_bratsgli_case_paths(path_or_idx: str | int):
    """Returns a dictionary like this:

    ```
    {
        "t1c": "path/to/t1c.nii.gz",
        "t1n": "path/to/t1n.nii.gz",
        "t2f": "path/to/t2f.nii.gz",
        "t2w": "path/to/t2w.nii.gz",
        "seg": "path/to/seg.nii.gz", # optional
    }
    ```

    Args:
        path_or_idx (str | int): Either path to the root or index of the study.
    """

    if isinstance(path_or_idx, int):
        path_or_idx = os.path.join(BRATS2024_GLI_ROOT, os.listdir(BRATS2024_GLI_ROOT)[path_or_idx])

    t1c = find_file_containing(path_or_idx, "t1c.", recursive=False)
    t1n = find_file_containing(path_or_idx, "t1n.", recursive=False)
    t2f = find_file_containing(path_or_idx, "t2f.", recursive=False)
    t2w = find_file_containing(path_or_idx, "t2w.", recursive=False)
    seg = find_file_containing(path_or_idx, "seg.", recursive=False, error=False)
    res = dict(t1c=t1c,t1n=t1n,t2f=t2f,t2w=t2w)
    if seg is not None: res['seg'] = seg
    return res