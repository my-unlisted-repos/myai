import csv
def csvread(path:str, delimiter=',', **kwargs) -> list[list[str]]:
    with open(path, 'r', encoding = 'utf8') as f:
        return list(csv.reader(f, delimiter=delimiter, **kwargs))

def csvwrite(path:str, data:list[list[str]], header: list[str] | None = None, delimiter=',', **kwargs) -> None:
    if header is not None:
        data.insert(0, header)
    with open(path, 'w', encoding = 'utf8') as f:
        csv.writer(f, delimiter=delimiter, **kwargs).writerows(data)
