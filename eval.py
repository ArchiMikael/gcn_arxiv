#########GRAPHS#########
import torch
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from spektral.data import Graph
from dgl import from_scipy
import numpy as np
from gcn_arxiv import *
import time

dataset = DglNodePropPredDataset(name="ogbn-arxiv")
out = dataset.num_classes

splitted_idx = dataset.get_idx_split()
tr_idx, va_idx, te_idx = (
    splitted_idx["train"],
    splitted_idx["valid"],
    splitted_idx["test"],
)
graph_dgl, labels_dgl = dataset[0]

srcs, dsts = graph_dgl.all_edges()
graph_dgl.add_edges(dsts, srcs)
graph_dgl = graph_dgl.remove_self_loop().add_self_loop()
n_classes = (labels_dgl.max() + 1).item()
graph_dgl.create_formats_()

N_DGL = graph_dgl.num_nodes()
F_DGL = graph_dgl.ndata["feat"].shape[1]

_x = np.array(graph_dgl.ndata['feat'])
_a = graph_dgl.adj_external(scipy_fmt='csr')
_y = np.array(labels_dgl)
graph_spektral = Graph(x=_x, a=_a, e=None, y=_y)

N_SP = graph_spektral.n_nodes
F_SP = graph_spektral.n_node_features

assert N_DGL == N_SP
assert F_DGL == F_SP

N = N_DGL
F = F_DGL
in_hidden = 256
out_hidden = 256

mask_tr = np.zeros(N, dtype=bool)
mask_va = np.zeros(N, dtype=bool)
mask_te = np.zeros(N, dtype=bool)
mask_tr[tr_idx] = True
mask_va[va_idx] = True
mask_te[te_idx] = True
masks = [mask_tr, mask_va, mask_te]

from spektral.transforms import AdjToSpTensor, GCNFilter
graph_spektral = GCNFilter()(graph_spektral)

graph_dgl = None
graph_dgl = from_scipy(graph_spektral.a, eweight_name="feat")
graph_dgl.edata["feat"] = graph_dgl.edata["feat"].float()
__x = torch.FloatTensor(graph_spektral.x)
graph_dgl.ndata["feat"] = __x

graph_spektral = AdjToSpTensor()(graph_spektral)
#########GRAPHS#########


##########INIT##########
evaluator = Evaluator(name="ogbn-arxiv")

loss_fn_spektral = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn_dgl = torch.nn.CrossEntropyLoss()
##########INIT##########


#########FUNCTS#########
def evaluate_spektral(model, graph, labels, masks, evaluator):
  tr_mask, va_mask, te_mask = masks

  pred = model([tf.convert_to_tensor(graph.x), graph.a], training=False)
  tr_loss = loss_fn_spektral(labels[tr_mask], pred[tr_mask])
  va_loss = loss_fn_spektral(labels[va_mask], pred[va_mask])
  te_loss = loss_fn_spektral(labels[te_mask], pred[te_mask])
  pred = pred.numpy().argmax(-1)[:, None]
  tr_acc = evaluator.eval({"y_true": labels[tr_mask], "y_pred": pred[tr_mask]})["acc"]
  va_acc = evaluator.eval({"y_true": labels[va_mask], "y_pred": pred[va_mask]})["acc"]
  te_acc = evaluator.eval({"y_true": labels[te_mask], "y_pred": pred[te_mask]})["acc"]
  return tr_acc, va_acc, te_acc, tr_loss, va_loss, te_loss
@torch.no_grad()
def evaluate_dgl(model, graph, labels, masks, evaluator):
  model.eval()
  feat = graph.ndata["feat"]
  tr_mask, va_mask, te_mask = masks

  pred = model(graph, feat)
  tr_loss = loss_fn_dgl(pred[tr_mask], labels[tr_mask][:, 0])
  va_loss = loss_fn_dgl(pred[va_mask], labels[va_mask][:, 0])
  te_loss = loss_fn_dgl(pred[te_mask], labels[te_mask][:, 0])
  pred = pred.argmax(dim=-1, keepdim=True)
  tr_acc = evaluator.eval({"y_pred": pred[tr_mask], "y_true": labels[tr_mask]})["acc"]
  va_acc = evaluator.eval({"y_pred": pred[va_mask], "y_true": labels[va_mask]})["acc"]
  te_acc = evaluator.eval({"y_pred": pred[te_mask], "y_true": labels[te_mask]})["acc"]

  return tr_acc, va_acc, te_acc, tr_loss, va_loss, te_loss
#########FUNCTS#########


##########LOAD##########
model_spektral = keras.models.load_model('./model_spektral.keras')
model_dgl = torch.load('./model_dgl.pt')
##########LOAD##########


##########EVAL##########
print("Evaluating model.")
tic = time.time()
tr_acc_dgl, va_acc_dgl, te_acc_dgl, tr_loss_dgl, va_loss_dgl, te_loss_dgl = evaluate_dgl(model_dgl, graph_dgl, labels_dgl, [tr_idx, va_idx, te_idx], evaluator)
toc = time.time()
eval_time = toc - tic
print(f"Done in {eval_time:.2f}! - Test acc: {te_acc_dgl:.8f}")

print("Evaluating model.")
tic = time.time()
tr_acc_spektral, va_acc_spektral, te_acc_spektral, tr_loss_spektral, va_loss_spektral, te_loss_spektral = evaluate_spektral(model_spektral, graph_spektral, graph_spektral.y, masks, evaluator)
toc = time.time()
eval_time = toc - tic
print(f"Done in {eval_time:.2f}! - Test acc: {te_acc_spektral:.8f}")
##########EVAL##########