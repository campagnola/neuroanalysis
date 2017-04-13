import weakref


class WeakRef(object):
    """Replacement for weakref that allows reference to None.
    """
    def __init__(self, obj):
        if obj is None:
            self._ref = None
        else:
            self._ref = weakref.ref(obj)
            
    def __call__(self):
        if self._ref is None:
            return None
        else:
            obj = self._ref()
            if obj is None:
                raise RuntimeError("Referenced object has already been collected.")
            return obj
        
    @property
    def is_dead(self):
        if self._ref is None:
            return False
        else:
            return self._ref() is None


def downsample(data, n, axis=0):
    """Downsample by averaging points together across axis.
    """
    n = int(n)
    if n == 1:
        return data
    elif n < 1:
        raise ValueError("Invalid downsampling window %d" % n)
    
    n_pts = int(data.shape[axis] / n)
    s = list(data.shape)
    s[axis] = n_pts
    s.insert(axis+1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, n_pts*n)
    d1 = data[tuple(sl)]
    d1.shape = tuple(s)
    d2 = d1.mean(axis+1)

    return d2
