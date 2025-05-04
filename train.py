#########GRAPHS#########
import torch
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from spektral.data import Graph
from dgl import from_scipy
import numpy as np
from gcn_arxiv import *

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
import torch.optim as optim
import time

model_spektral = GCN_spektral(F, n_classes, N, 256, 0.75)
optimizer_spektral = keras.optimizers.Adam(learning_rate=0.005)
model_spektral.optimizer = optimizer_spektral
lr_scheduler_spektral = keras.callbacks.ReduceLROnPlateau(
    mode="min",
    factor=0.5,
    patience=100,
    min_lr=1e-5,
)
lr_scheduler_spektral.set_model(model_spektral)

model_dgl = GCN_dgl(F, n_classes, N, 256, 0.75)
model_dgl = model_dgl.to(torch.device("cpu"))
optimizer_dgl = optim.Adam(
    model_dgl.parameters(), lr=0.005
)
lr_scheduler_dgl = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_dgl,
    mode="min",
    factor=0.5,
    patience=100,
    min_lr=1e-5,
)

evaluator = Evaluator(name="ogbn-arxiv")

loss_fn_spektral = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn_dgl = torch.nn.CrossEntropyLoss()
##########INIT##########


#########FUNCTS#########
def adjust_learning_rate_spektral(optimizer, lr, epoch):
  if epoch <= 50:
    optimizer.learning_rate.assign(lr * epoch / 50)
def adjust_learning_rate_dgl(optimizer, lr, epoch):
  if epoch <= 50:
      for param_group in optimizer.param_groups:
          param_group["lr"] = lr * epoch / 50

def train_spektral(model, graph, labels, train_idx, optimizer):
  with tf.GradientTape() as tape:
    pred = model([tf.convert_to_tensor(graph.x), graph.a], training=True)
    loss = loss_fn_spektral(labels, pred)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, pred
def train_dgl(model, graph, labels, train_idx, optimizer):
    model.train()
    feat = graph.ndata["feat"]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = loss_fn_dgl(pred, labels[:, 0])
    loss.backward()
    optimizer.step()

    return loss, pred

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


########TRAINING########
n_epochs = 1000

total_time_dgl = 0
total_time_spektral = 0
best_va_acc_dgl, final_te_acc_dgl, best_va_loss_dgl = 0, 0, float("inf")
best_va_acc_spektral, final_te_acc_spektral, best_va_loss_spektral = 0, 0, float("inf")

for epoch in range(1, n_epochs + 1):
  SEED = epoch

  adjust_learning_rate_dgl(optimizer_dgl, 0.005, epoch)
  adjust_learning_rate_spektral(optimizer_spektral, 0.005, epoch)

  tic_dgl = time.time()
  loss_dgl, pred_dgl = train_dgl(model_dgl, graph_dgl, labels_dgl, tr_idx, optimizer_dgl)
  _pred_dgl = pred_dgl.argmax(dim=-1, keepdim=True)
  acc_dgl = evaluator.eval({"y_pred": _pred_dgl, "y_true": labels_dgl})["acc"]
  tr_acc_dgl, va_acc_dgl, te_acc_dgl, tr_loss_dgl, va_loss_dgl, te_loss_dgl = evaluate_dgl(model_dgl, graph_dgl, labels_dgl, [tr_idx, va_idx, te_idx], evaluator)
  lr_scheduler_dgl.step(loss_dgl)
  toc_dgl = time.time()
  total_time_dgl += toc_dgl - tic_dgl

  tic_spektral = time.time()
  loss_spektral, pred_spektral = train_spektral(model_spektral, graph_spektral, graph_spektral.y, mask_tr, optimizer_spektral)
  _pred_spektral = pred_spektral.numpy().argmax(-1)[:, None]
  acc_spektral = evaluator.eval({"y_pred": _pred_spektral, "y_true": graph_spektral.y})["acc"]
  tr_acc_spektral, va_acc_spektral, te_acc_spektral, tr_loss_spektral, va_loss_spektral, te_loss_spektral = evaluate_spektral(model_spektral, graph_spektral, graph_spektral.y, masks, evaluator)
  lr_scheduler_spektral.on_epoch_end(epoch, {'val_loss': loss_spektral})
  toc_spektral = time.time()
  total_time_spektral += toc_spektral - tic_spektral

  #model_dgl.eval()
  #pred_dgl = model_dgl(graph_dgl, graph_dgl.ndata["feat"]).detach().numpy()
  #pred_spektral = model_spektral([tf.convert_to_tensor(graph_spektral.x), graph_spektral.a], training=False).numpy()
  #discr = pred_spektral - pred_dgl
  #discr_dgl = max([abs(item) for item in (discr).ravel()])/max([abs(item) for item in (pred_dgl).ravel()])
  #discr_spektral = max([abs(item) for item in (discr).ravel()])/max([abs(item) for item in (pred_spektral).ravel()])
  #print("-------------------------" + str(epoch) + "-------------------------")
  #print("Discrepancy (max in DGL): " + str(100.0*discr_dgl) + "%")
  #print("Discrepancy (max in Spektral): " + str(100.0*discr_spektral) + "%")
  #print("-------------------------" + str(epoch) + "-------------------------")

  if va_loss_dgl < best_va_loss_dgl:
    best_va_loss_dgl = va_loss_dgl
    best_va_acc_dgl = va_acc_dgl
    final_te_acc_dgl = te_acc_dgl

  if va_loss_spektral < best_va_loss_spektral:
    best_va_loss_spektral = va_loss_spektral
    best_va_acc_spektral = va_acc_spektral
    final_te_acc_spektral = te_acc_spektral

  if epoch % 10 == 0:
    print()
    print(
      "----------------------------\n"
      f"Epoch: {epoch}/{n_epochs}\n"
      "-------------DGL------------\n"
      f"Average epoch time: {total_time_dgl / epoch:.2f}\n"
      f"Loss: {loss_dgl.item():.4f}, Acc: {acc_dgl:.4f}\n"
      f"Train/Val/Test loss: {tr_loss_dgl:.4f}/{va_loss_dgl:.4f}/{te_loss_dgl:.4f}\n"
      f"Train/Val/Test/Best val/Final test acc: {tr_acc_dgl:.4f}/{va_acc_dgl:.4f}/{te_acc_dgl:.4f}/{best_va_acc_dgl:.4f}/{final_te_acc_dgl:.4f}\n"
      "----------Spektral----------\n"
      f"Average epoch time: {total_time_spektral / epoch:.2f}\n"
      f"Loss: {loss_spektral:.4f}, Acc: {acc_spektral:.4f}\n"
      f"Train/Val/Test loss: {tr_loss_spektral:.4f}/{va_loss_spektral:.4f}/{te_loss_spektral:.4f}\n"
      f"Train/Val/Test/Best val/Final test acc: {tr_acc_spektral:.4f}/{va_acc_spektral:.4f}/{te_acc_spektral:.4f}/{best_va_acc_spektral:.4f}/{final_te_acc_spektral:.4f}\n"
      "----------------------------\n"
    )
########TRAINING########


#########SAVING#########
model_spektral.save('./model_spektral.keras')
torch.save(model_dgl, './model_dgl.pt')
#########SAVING#########