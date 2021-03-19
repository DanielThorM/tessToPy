import numpy as np
class absdict(dict):
    def __setitem__(self, key, item):
        if isinstance(key, int):
            if np.sign(key) == -1:
                raise Exception ('Can not assign negative keys')
            else:
                self.__dict__[key] = item
        else:
            self.__dict__[key] = item

    def __getitem__(self, key):
        if isinstance(key, int):
            if np.sign(key) == -1:
                if isinstance(self.__dict__[abs(key)], list):
                    return self.__dict__[abs(key)][::-1]
                else:
                    return self.__dict__[abs(key)].reverse()
            else:
                return self.__dict__[key]
        else:
            return self.__dict__[key]


    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        if isinstance(key, int):
            del self.__dict__[abs(key)]
        else:
            del self.__dict__[key]


    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        if isinstance(k, int):
            return abs(k) in self.__dict__
        else:
            return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)