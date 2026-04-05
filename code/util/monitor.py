__all__ = ['AverageMeter']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='%.6f'):
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        s = self.fmt % self.avg
        return s
