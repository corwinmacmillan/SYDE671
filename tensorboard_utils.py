import os
import numpy as np
import matplotlib.pyplot as plt

def inspect_model(writer, model, loader, device):
    '''
    Tensorboard utiliy to show model
    :params:
        writer: tensorboard summary writer
        model: model 
        loader: dataloader of model being trained for inputs
        device: device
    '''
    data_iter = iter(loader)
    inputs, labels = next(data_iter)
    inputs, labels = (inputs.to(device), labels.to(device))
    writer.add_graph(model, inputs)
    writer.close()

