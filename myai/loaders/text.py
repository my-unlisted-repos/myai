def txtread(path:str):
    with open(path, 'r', encoding = 'utf8') as f: return f.read()

def txtwrite(fname:str, s: str, ):
    with open(fname, 'w', encoding = 'utf8') as f: f.write(s)