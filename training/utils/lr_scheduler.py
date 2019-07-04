class MultiStepLR(object):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
    """
    """
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)     
    """


    def __init__(self, optimizer, milestones, gamma=0.1):
        self.param_group = []
        self.milestones = milestones
        self.gamma = gamma
        self.len = len(milestones)

        for param_group in optimizer.param_groups:
            self.param_group.append({'lr': param_group['lr']})

    def __call__(self, epoch, optimizer):
        decay = self.get_decay(epoch)
        for self_param_group, param_group in zip(self.param_group, optimizer.param_groups):
            param_group['lr'] = self_param_group['lr']*self.gamma**decay

        return optimizer

    def get_decay(self, epoch):
        for decay, milestone in zip(range(self.len), self.milestones):
            if epoch <= milestone:
                return decay

        return decay + 1

