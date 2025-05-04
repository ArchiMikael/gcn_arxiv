##########VARS##########
EPS = 1e-5
MOMENTUM = 0.99
SEED = 0
##########VARS##########


#########MODELS#########
import tensorflow as tf
from tensorflow import keras
from keras import saving
import torch
from spektral.layers import GCNConv as GCNConv_SP
from keras.layers import BatchNormalization as BatchNorm_SP
from keras.activations import relu as ReLU_SP
from dgl.nn.pytorch import GraphConv as GCNConv_DGL
from torch.nn import BatchNorm1d as BatchNorm_DGL
from torch.nn.functional import relu as ReLU_DGL
from torch.nn import init
import numpy as np

@saving.register_keras_serializable()
class Dropout_TF(keras.layers.Layer):
  def __init__(self, rate, offset=0):
    super().__init__()
    self.rate = rate
    self.generator = np.random.uniform
    self.offset = offset

  def call(self, inputs, training=False):
    if not training or self.rate == 0:
      return inputs
    np.random.seed(SEED + self.offset)
    mask = tf.convert_to_tensor(self.generator(size=inputs.shape.as_list()) > self.rate, tf.float32)
    return inputs * mask / (1 - self.rate)

class Dropout_PT(torch.nn.Module):
  def __init__(self, rate, offset=0):
    super().__init__()
    self.rate = rate
    self.generator = np.random.uniform
    self.offset = offset

  def forward(self, inputs):
    if not self.training or self.rate == 0:
      return inputs
    np.random.seed(SEED + self.offset)
    mask = torch.from_numpy(self.generator(size=inputs.shape) > self.rate)
    return inputs * mask / (1 - self.rate)

@saving.register_keras_serializable()
class GCN_spektral(keras.models.Model):
  def __init__(self, F, n_classes, N, channels, dropout, **kwrgs):
    super().__init__()
    self.F = F
    self.n_classes = n_classes
    self.N = N
    self.channels = channels
    self.dropout = dropout

    #Dropout = keras.layers.Dropout
    Dropout = Dropout_TF

    self.input_drop = Dropout(min(0.1, dropout))

    self.GCNConv_1 = GCNConv_SP(channels, activation=None, bias=False, kernel_initializer=keras.initializers.Constant(0.5))
    self.BatchNorm_1 = BatchNorm_SP(epsilon=EPS, momentum=MOMENTUM, beta_initializer="zeros", gamma_initializer="ones")
    self.activation_1 = ReLU_SP
    self.drop_1 = Dropout(dropout, offset=1)

    self.GCNConv_2 = GCNConv_SP(channels, activation=None, bias=False, kernel_initializer=keras.initializers.Constant(0.5))
    self.BatchNorm_2 = BatchNorm_SP(epsilon=EPS, momentum=MOMENTUM, beta_initializer="zeros", gamma_initializer="ones")
    self.activation_2 = ReLU_SP
    self.drop_2 = Dropout(dropout, offset=2)

    self.GCNConv_3 = GCNConv_SP(n_classes, activation=None, bias=True, kernel_initializer=keras.initializers.Constant(0.5), bias_initializer=keras.initializers.Constant(0.5))

  def call(self, inputs, training=False):
    h, a = inputs
    h = self.input_drop(h)

    h = self.GCNConv_1([h, a], mask=[1])
    h = self.BatchNorm_1(h, training=training)
    h = self.activation_1(h)
    h = self.drop_1(h, training=training)

    h = self.GCNConv_2([h, a], mask=[1])
    h = self.BatchNorm_2(h, training=training)
    h = self.activation_2(h)
    h = self.drop_2(h, training=training)

    h = self.GCNConv_3([h, a], mask=[1])

    return h
  
  def get_config(self):
    config = super().get_config()
        
    config.update(
      {
        "F": self.F,
        "n_classes": self.n_classes,
        "N": self.N,
        "channels": self.channels,
        "dropout": self.dropout,
      }
    )

    return config

class GCN_dgl(torch.nn.Module):
  def __init__(self, F, n_classes, N, channels, dropout, **kwrgs):
    super().__init__()

    #Dropout = torch.nn.Dropout
    Dropout = Dropout_PT

    self.input_drop = Dropout(min(0.1, dropout))

    self.GCNConv_1 = GCNConv_DGL(F, channels, "none", bias=False)
    self.GCNConv_1.weight.data.fill_(0.5)
    self.BatchNorm_1 = BatchNorm_DGL(channels, eps=EPS, momentum=1-MOMENTUM)
    init.ones_(self.BatchNorm_1.weight)
    init.zeros_(self.BatchNorm_1.bias)
    self.activation_1 = ReLU_DGL
    self.drop_1 = Dropout(dropout, offset=1)

    self.GCNConv_2 = GCNConv_DGL(channels, channels, "none", bias=False)
    self.GCNConv_2.weight.data.fill_(0.5)
    self.BatchNorm_2 = BatchNorm_DGL(channels, eps=EPS, momentum=1-MOMENTUM)
    init.ones_(self.BatchNorm_2.weight)
    init.zeros_(self.BatchNorm_2.bias)
    self.activation_2 = ReLU_DGL
    self.drop_2 = Dropout(dropout, offset=2)

    self.GCNConv_3 = GCNConv_DGL(channels, n_classes, "none", bias=True)
    self.GCNConv_3.weight.data.fill_(0.5)
    self.GCNConv_3.bias.data.fill_(0.5)

  def forward(self, graph, feat):
    h = feat
    h = self.input_drop(h)

    h = self.GCNConv_1(graph, h, edge_weight=graph.edata["feat"])
    h = self.BatchNorm_1(h)
    h = self.activation_1(h)
    h = self.drop_1(h)

    h = self.GCNConv_2(graph, h, edge_weight=graph.edata["feat"])
    h = self.BatchNorm_2(h)
    h = self.activation_2(h)
    h = self.drop_2(h)

    h = self.GCNConv_3(graph, h, edge_weight=graph.edata["feat"])

    return h
#########MODELS#########