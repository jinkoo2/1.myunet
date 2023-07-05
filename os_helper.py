def write_all_text(path, txt):
    f = open(path, "w")
    f.write(txt)
    f.close()

def read_all_text(path):
    f = open(path, "r")
    txt = f.read()
    f.close()
    return txt