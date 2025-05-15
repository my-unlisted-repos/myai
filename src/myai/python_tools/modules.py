import os

# ---------------------------------------------------------------------------- #
#                                     sada                                     #
# ---------------------------------------------------------------------------- #

def _add_frame(s: str):
    frame = "# ---------------------------------------------------------------------------- #"
    return f"{frame}\n#{s:^78}#\n{frame}"


def module_to_text(path):
    """path is path to folder where __init__ is."""
    dirname = os.path.dirname(path)

    def _read(file):
        with open(file, 'r', encoding='utf8') as f: return f.read()

    text = ''
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                text = f'{text}\n\n{_add_frame(file_path.replace(dirname,""))}\n\n{_read(file_path)}'

    return text[2:]
