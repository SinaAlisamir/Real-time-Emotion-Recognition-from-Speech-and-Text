import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class myMultiLin(nn.Module):
    def __init__(self, featSize, outputSizes=[1]):
        super(myMultiLin, self).__init__()

        self.featSize = featSize
        self.outputSizes = outputSizes

        self.linears = nn.ModuleList([nn.Linear(featSize, outputSizes[i]) for i in range(len(outputSizes))])

    def forward(self, x, taskID=0):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        # output = output[:, -1, :].unsqueeze(1)
        output = self.linears[taskID](x)
        # output = self.lin(output)
        return output
        
class myMultiGRU(nn.Module):
    def __init__(self, featSize, hidden_size=128, num_layers=1, outputSizes=[1]):
        super(myMultiGRU, self).__init__()

        self.featSize = featSize
        self.outputSizes = outputSizes

        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, outputSizes[i]) for i in range(len(outputSizes))])

    def forward(self, x, taskID=0):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        # output = output[:, -1, :].unsqueeze(1)
        output = self.linears[taskID](output)
        # output = self.lin(output)
        return output
        
class myGRU(nn.Module):
    def __init__(self, featSize, hidden_size=128, num_layers=1, outputSize=1):
        super(myGRU, self).__init__()

        self.featSize = featSize
        self.outputSize = outputSize

        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.last = nn.Linear(hidden_size, outputSize)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        # output = output[:, -1, :].unsqueeze(1)
        output = self.last(output)
        output = self.tanh(output)
        # output = self.lin(output)
        return output

class myTransformer(nn.Module):
    def __init__(self, featSize, hidden_size=128, nhead=8, num_layers=1, outputSize=1):
        super(myTransformer, self).__init__()

        self.featSize = featSize
        self.outputSize = outputSize
        self.pos_encoder = PositionalEncoding(featSize)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=featSize, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.last = nn.Linear(featSize, outputSize)
        self.tanh = nn.Tanh()
        # self.tanh = nn.Tanh()
        # self.logsoft = nn.LogSoftmax(1)

    def forward(self, x):
        # output = x.transpose(1,0)
        x = x * math.sqrt(self.featSize)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = self.last(output)
        output = self.tanh(output)
        # if self.tasks[taskID] == 0:
        #     output = self.tanh(output)
        # if self.tasks[taskID] == 1:
        #     output = output[:, 0, :] # last only
        #     output = self.logsoft(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 7000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(1,0)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(1,0)
        return x


class myMultiModalGRU(nn.Module):
    def __init__(self, featSizeA, featSizeL, fusion="cat", L_size=1024, pool_L=False, hidden_size=128, num_layers=1, outputSizes=[1]):
        super().__init__()

        self.featSizeA = featSizeA
        self.featSizeL = featSizeL
        self.fusion = fusion
        self.hidden_size = hidden_size
        self.outputSizes = outputSizes
        self.pool_L = pool_L
        self.L_size = L_size

        self.rnn_A = nn.GRU(
            input_size=featSizeA, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)

        self.rnn_L = nn.GRU(
            input_size=featSizeL, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        
        linear_hidden_size = 2*hidden_size if "cat" in fusion else hidden_size

        if "A_L" in fusion:
            if "A+A_L" in fusion:
                linear_hidden_size = 2*hidden_size + L_size
            elif "cat" in fusion:
                linear_hidden_size = hidden_size + L_size
            else:
                linear_hidden_size = L_size
        # if "L" in type:
        #     if (not "cat" in fusion) and (not "A" in fusion):
        #         linear_hidden_size = hidden_size
        if self.pool_L:
            linear_hidden_size = linear_hidden_size - hidden_size + L_size
        # print("linear_hidden_size", linear_hidden_size)
        
        self.linears = nn.ModuleList([nn.Linear(linear_hidden_size, outputSizes[i]) for i in range(len(outputSizes))])

    def add_position(self, x):
        output = x * math.sqrt(self.hidden_size)
        output = self.pos_encoder(output)
        return output

    def forward(self, x_A=None, x_L=None, x_A_L=None, taskID=0):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        if not x_A is None:
            output_A, _ = self.rnn_A(x_A)
        if not x_L is None:
            output_L, _ = self.rnn_L(x_L)
        if not x_A is None:
            output_A = output_A[:, -1, :].unsqueeze(1)
            output = output_A
        if not x_L is None:
            output_L = output_L[:, -1, :].unsqueeze(1)
            output = output_L
        if "A_L" in self.fusion:
            if "A+A_L" in self.fusion:
                output_A = torch.cat([output_A, x_A_L], dim=2)
            else:
                output_A = x_A_L
        if self.pool_L:
            output_L = x_L
        # print("output_A", output_A.size())
    
        if (not x_A is None) and (not x_L is None): # both x_A and x_L should exist to fuse
            if "cat" in self.fusion: 
                output = torch.cat([output_A, output_L], dim=2)
            else:
                output = (output_A + output_L) / 2
        # output = output[:, -1, :].unsqueeze(1)
        if x_A is None: output = output_L
        if x_L is None: output = output_A
        # print("output", output.size())
        output = self.linears[taskID](output)
        # output = self.lin(output)
        return output