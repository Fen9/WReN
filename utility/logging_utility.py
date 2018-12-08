import os
import numpy as np

import seaborn as sns
from tensorboardX import SummaryWriter

class logwrapper(object):
    def __init__(self, log_path):
        self.writter = SummaryWriter(log_path)
    
    def write_scalar(self, name, scalar, epoch):
        self.writter.add_scalar(name, scalar, epoch)
    
    def write_scalars(self, name, scalars, epoch):
        self.writter.add_scalars(name, scalars, epoch)

    def write_text(self, title, content):
        self.writter.add_text(title, content)
    
    def close_wrapper(self):
        self.writter.close()


class plotwrapper(object):
    def __init__(self, plot_path):
        self.plot_path = plot_path

    