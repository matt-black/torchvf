

class EarlyStopper(object):
    def __init__(self, patience, min_delta):
        self._patience  = patience
        self._min_delta = min_delta
        self._count     = 0
        self._min_loss  = float('inf')

    def stop_early(self, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self._count = 0
        elif loss > (self._min_loss + self._min_delta):
            self._count += 1
            if self._count >= self._patience:
                return True
        else:
            pass
        return False
