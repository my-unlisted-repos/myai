import os
import shutil
from collections import OrderedDict
from collections.abc import Callable, Sequence

from ..python_tools import get_all_files


def flatten_dir(
    root: str,
    outdir: str,
    extensions: str | Sequence[str] | None = None,
    path_filter: Callable[[str], bool] | None = None,
    sep: str | None = ".",
    ext_override: str | None = None,
    combine_small: int | None = None,
):
    """Copy all files in a directory into another directory, flattened.

    Args:
        root (str): directory to look in.
        outdir (str): output directory
        extensions (str | Sequence[str] | None, optional): filter by extension (case insensitive). Defaults to None.
        path_filter (_type_, optional): _description_. Defaults to None.
        sep (str | None, optional): if not None, includes path to the file in the file name, with `/` replaced by sep. Defaults to '.'.
        ext_override (str | None, optional): if not None, replaces extensions of output files with this. Shouldn't include `.` in it. Defaults to None.
        combine_small (str | None, optional): integer of maximum number of files, will combines smallest total modules into one file until this is met. Defaults to None.
    """
    root = os.path.normpath(root)
    files = get_all_files(root, extensions = extensions, path_filter=path_filter)

    if combine_small is None:
        for f in files:
            if sep is not None: name = f.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
            else: name = os.path.basename(f)

            if ext_override is not None:
                if '.' in name: name = '.'.join(name.split('.')[:-1])
                name = name + f'.{ext_override}'

            shutil.copyfile(f, os.path.join(outdir, name))

        return

    # now we have to combine smallest modules
    def read(f):
        try:
            with open(f, 'r', encoding='utf8') as file: return file.read()
        except UnicodeDecodeError:
            print(f)
            with open(f, 'r', encoding='Windows-1252') as file: return file.read()

    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None:
        extensions = [i if i.startswith('.') else f'.{i}' for i in extensions]
    branches = {}

    def check_file(f):
        if extensions is None: ext = True
        else: ext = f.lower().endswith(tuple(extensions))

        if path_filter is None: pp = True
        else: pp = path_filter(f)
        return ext and pp and os.path.isfile(f)

    # dirs with files in them
    for r, dirs, fs in os.walk(root):
        full_files = [os.path.join(r, f) for f in fs]
        if any(os.path.isfile(f) and check_file(f) for f in full_files):
            branches[r] = sum(len(read(f)) for f in full_files if check_file(f))

    branches_t = sorted(branches.items(), key = lambda x: x[1])

    files_to_save: dict[str, str|None] = {f: None for f in files if check_file(f)}

    renamed = {}
    while len(files_to_save) > combine_small:
        if len(branches_t) == 0:
            break

        rr = branches_t.pop(0)
        r = rr[0]

        text = ''
        if sep is not None: name = r.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
        else: name = os.path.basename(r)

        if ext_override is not None:
            if '.' in name: name = '.'.join(name.split('.')[:-1])
            if len(name) == 0: name = 'root'
            name = name + f'.{ext_override}'

        texts = [(os.path.join(r, f), read(os.path.join(r, f))) for f in os.listdir(r) if check_file(os.path.join(r, f))]

        for f, ftext in sorted(texts, key = lambda x: len(x[1])):
            del files_to_save[f]
            if len(text) == 0: text = f'# ---- {f.replace(root, '')} -------\n\n{ftext}'
            else: text += f'\n\n# ---- {f.replace(root, '')} -------\n\n{ftext}'

            if len(files_to_save) + 1 <= combine_small: break

        files_to_save[name] = text
        renamed[name] = r


    while len(files_to_save) > combine_small:

        files_loaded = sorted([(f, t) if t is not None else (f, read(f)) for f,t in files_to_save.items()], key=lambda x: len(x[1]))
        files_lodadedd = dict(files_loaded)
        import difflib
        smallest, t1 = files_loaded.pop(0)
        match = difflib.get_close_matches(smallest, [i[0] for i in files_loaded], n = 1, cutoff=0)[0]
        t2 = files_lodadedd[match]

        del files_to_save[smallest],files_to_save[match]

        if smallest in renamed: smallest = renamed[smallest]
        if match in renamed: match = renamed[match]

        print(smallest)
        text = f'# ---- {smallest.replace(root, '')} -------\n\n{t1}'
        text += f'\n\n# ---- {match.replace(root, '')} -------\n\n{t2}'

        n1 = os.path.basename(smallest)
        n2 = os.path.basename(match)

        dn1 = os.path.dirname(smallest)
        dn2 = os.path.dirname(match)

        if len(dn1) < len(dn2):
            name = f'{dn1}/{n1}+{n2}'
        else:
            name = f'{dn2}/{n2}+{n1}'

        while name in files_to_save:
            import random
            name += str(random.randint(0,9))

        files_to_save[name] = text

    for f, t in files_to_save.items():
        if '/' in f or '\\' in f:
            if sep is not None: name = f.replace(root, '').replace('/', sep).replace('\\', sep)[1:]
            else: name = os.path.basename(f)

            if ext_override is not None:
                if '.' in name: name = '.'.join(name.split('.')[:-1])
                name = name + f'.{ext_override}'
        else:
            name = f

        with open(os.path.join(outdir, name), 'w', encoding = 'utf8') as file:
            if t is None: t = read(f)
            file.write(t)

def flatten_dir_flat(
    root: str,
    outdir: str,
    extensions: str | Sequence[str] | None = None,
    path_filter: Callable[[str], bool] | None = None,
    text_filter: Callable[[str], bool] | None = None,
    sep: str | None = ".",
    ext_override: str | None = None,
    combine_small: int | None = None,
    mkdir = True
):
    def read(f):
        try:
            with open(f, 'r', encoding='utf8') as file: return file.read()
        except UnicodeDecodeError:
            print(f)
            with open(f, 'r', encoding='Windows-1252') as file: return file.read()

    files = get_all_files(root, extensions=extensions, path_filter=path_filter)
    files_content: OrderedDict[tuple[str, ...], str] = OrderedDict(sorted([((path,), read(path)) for path in files], key = lambda x: len(x[1])))

    if text_filter is not None:
        files_content = OrderedDict([(k,v) for k, v in files_content.items() if text_filter(v)])

    if combine_small is None: combine_small = 999999999
    while len(files_content) > combine_small:
        key1  =list(files_content.keys())[0]
        key2 = list(files_content.keys())[1]
        text1 = files_content.pop(key1)
        text2 = files_content.pop(key2)

        files_content[key1 + key2] = f'#-----------------{key1}-------------\n{text1}\n\n#-----------------{key2}-------------\n{text2}'


    if mkdir and not os.path.exists(outdir): os.mkdir(outdir)

    def override(fname):
        if '.' in fname: fname = '.'.join(fname.split('.')[:-1])
        fname = f'{fname}.{ext_override}'
        return fname

    saved_fnames = set()
    if sep is None: sep = ''
    for files, text in files_content.items():
        fnames = [os.path.basename(i) for i in files]

        def remove_ext(fname):
            if '.' in fname: return  '.'.join(fname.split('.')[:-1])
            return fname

        if len(fnames) > 1:
            fnames = list(map(remove_ext, fnames[:-1])) + [fnames[-1]]

        while sum(map(len, fnames)) > 200:
            if len((fnames)) >= 199: fnames = fnames[:199]
            longest_fname = sorted(fnames, key=len)[-1]
            fnames[fnames.index(longest_fname)] = fnames[fnames.index(longest_fname)][:-1]

        filename = sep.join(fnames)
        i = 1
        while filename in saved_fnames:
            i+=1
            b = filename
            filename = f'{filename} ({i})'
            if filename in saved_fnames: filename = b

        if ext_override is not None:
            filename = override(filename)

        saved_fnames.add(filename)
        with open(os.path.join(outdir, filename), 'w', encoding='utf8') as f:
            f.write(text)

