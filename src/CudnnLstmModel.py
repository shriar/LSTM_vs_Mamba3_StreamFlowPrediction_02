"""A class for an LSTM model that uses Cuda"""
from hydroDL.model.rnn.CudnnLstm import CudnnLstm
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn
import csv
import numpy as n


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, nLayer=1, dr=0.5, warmUpDay=None):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = nLayer
        self.linearIn = torch.nn.Linear(nx, hiddenSize)

        # Stack HydroDL CudnnLstm cells in ModuleList for multi-layer support
        self.lstm_layers = torch.nn.ModuleList([
            rnn.CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
            for _ in range(nLayer)
        ])

        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.name = "CudnnLstmModel"
        self.is_legacy = True
        self.warmUpDay = warmUpDay

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """

        :param inputs: a dictionary of input data (x and potentially z data)
        :param doDropMC:
        :param dropoutFalse:
        :return:
        """
        # if not self.warmUpDay is None:
        #     x, warmUpDay = self.extend_day(x, warm_up_day=self.warmUpDay)

        h = F.relu(self.linearIn(x))

        # Pass through each LSTM layer sequentially
        for cell in self.lstm_layers:
            h, _ = cell(h, doDropMC=doDropMC, dropoutFalse=dropoutFalse)

        out = self.linearOut(h)

        # if not self.warmUpDay is None:
        #     out = self.reduce_day(out, warm_up_day=self.warmUpDay)

        return out

    def extend_day(self, x, warm_up_day):
        x_num_day = x.shape[0]
        warm_up_day = min(x_num_day, warm_up_day)
        x_select = x[:warm_up_day, :, :]
        x = torch.cat([x_select, x], dim=0)
        return x, warm_up_day

    def reduce_day(self, x, warm_up_day):
        x = x[warm_up_day:,:,:]
        return x
