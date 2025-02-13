def txtread(path:str):
    with open(path, 'r', encoding = 'utf8') as f: return f.read()

def txtwrite(s: str, path: str):
    with open(path, 'w', encoding = 'utf8') as f: f.write(s)