import os

def split_directory_path(path):
    dir_names = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            dir_names.append(folder)
        else:
            if path != "":
                dir_names.append(path)
            break

    dir_names.reverse()
    return dir_names

file_path="../path/to/your/file.txt"

list = split_directory_path(file_path)
print(list)
print(list[len(list)-2])


