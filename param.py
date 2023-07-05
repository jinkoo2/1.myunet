import json


class Param(dict):
    def __init__(self, *arg, **kw):
        super(Param, self).__init__(*arg, **kw)

    def load_from_txt(self, param_file):
        self.clear()
        with open(param_file) as myfile:
            for line in myfile:
                line = line.strip()

                if line == '':
                    continue

                if line[0] == '#':
                    continue

                key, value = line.partition('=')[::2]
                self[key.strip()] = value.strip()

    def has_key(self, key):
        return key in self.keys()

    def get_item_as_float_array(self, key, delim):
        string_array = self[key].split(delim)
        return [float(x) for x in string_array]

    def get_item_as_int_array(self, key, delim):
        string_array = self[key].split(delim)
        return [int(x) for x in string_array]

    def cast_to_int(self, key):
        if self.has_key(key):
            self[key] = int(self[key])

    def cast_to_bool(self, key):
        if self.has_key(key):
            self[key] = bool(self[key])

    def cast_to_float(self, key):
        if self.has_key(key):
            self[key] = float(self[key])

    def cast_to_int_array(self, key, delim=','):
        if self.has_key(key):
            self[key] = self.get_item_as_int_array(key, delim)

    def cast_to_float_array(self, key, delim=','):
        if self.has_key(key):
            self[key] = self.get_item_as_float_array(key, delim)

    def save_to_txt(self, param_file):
        file = open(param_file, "w")
        for key in self:
            file.write(f'{key}={self[key]}\n')
        file.close()

    def load_from_json(self, json_file):
        self.clear()
        # read txt file
        file = open(json_file, 'r')
        txt = file.read()
        file.close()

        # string to object
        dict = json.loads(txt)

        # copy dictionary to my self
        for key in dict:
            self[key] = dict[key]

    def save_to_json(self, json_file):
        json_str = json.dumps(self)
        file = open(json_file, "w")
        file.write(json_str)
        file.close()

# helper function


def param_from_txt(txt_file):
    p = Param()
    p.load_from_txt(txt_file)
    return p


def param_from_json(json_file):
    p = Param()
    p.load_from_json(json_file)
    return p
