from importlib import import_module
# import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler


def getGradualWarmupScheduler(optimizer, multiplier, warmup_epochs, after_scheduler_conf):
    """
    wrapper of GradualWarmupScheduler (in warmup_scheduler / pytorch-gradual-warmup-lr)

    after_scheduler_conf: {'type':xx, 'args':yy}, args excluding optimizer, total_epochs
    """
    # instantiate after_scheduler
    modulename, clsname = after_scheduler_conf['type'].rsplit('.', 1)
    mod = import_module(modulename)
    scheduler_cls = getattr(mod, clsname)
    after_scheduler = scheduler_cls(optimizer, **after_scheduler_conf['args'])

    # instantiate GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=after_scheduler)

    return scheduler
