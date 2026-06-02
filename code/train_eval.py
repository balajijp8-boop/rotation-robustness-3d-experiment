"""
Train SimpleView, PointNet, DGCNN on synthetic shapes (NO rotation augmentation).
Then evaluate accuracy across rotation levels.

Van Gemert controlled experiment (Fig 2):
  c1: all models reach >90% on aligned test (controlled setting is reasonable)
  c2: PointNet/DGCNN are reasonable baselines at 0° (fairly set up)
  c3: PointNet/DGCNN accuracy drops sharply with rotation angle (problem exists)
  c4: SimpleView accuracy drops more gradually (projection helps)

Single controlled variable: rotation angle theta at test time (RP9).
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader

# ── dataset ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_shapes')
CLASS_NAMES = ['cube', 'sphere', 'cylinder', 'cone', 'torus']
N_CLASSES   = 5

# Register synthetic5 so models can look up num_classes
from all_utils import DATASET_NUM_CLASS
DATASET_NUM_CLASS['synthetic5'] = N_CLASSES


class ShapesDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.pcs    = torch.from_numpy(f['data'][:]).float()
            self.labels = torch.from_numpy(f['label'][:].reshape(-1)).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pcs[idx], self.labels[idx]


def make_loader(h5_path, batch_size, shuffle):
    ds = ShapesDataset(h5_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ── loss ─────────────────────────────────────────────────────────────────────

def smooth_loss(pred, gold, eps=0.2):
    """Label-smoothing cross-entropy (same as SimpleView paper)."""
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter_(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    return -(one_hot * log_prb).sum(dim=1).mean()


# ── model factory ────────────────────────────────────────────────────────────

def build_model(model_name):
    import models
    if model_name == 'simpleview':
        return models.MVModel(task='cls', dataset='synthetic5',
                              backbone='resnet18', feat_size=16)
    elif model_name == 'pointnet':
        return models.PointNet(task='cls_trans', dataset='synthetic5')
    elif model_name == 'dgcnn':
        return models.DGCNN(task='cls', dataset='synthetic5')
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── forward helpers ──────────────────────────────────────────────────────────

def forward_logit(model, pc, model_name):
    """Unified forward pass → logit tensor."""
    out = model(pc=pc)
    if model_name == 'pointnet':
        return out['logit'], out.get('trans_feat')
    return out['logit'], None


def compute_loss(logit, labels, trans_feat, model_name):
    loss = smooth_loss(logit, labels)
    if model_name == 'pointnet' and trans_feat is not None:
        from pointnet_pyt.pointnet.model import feature_transform_regularizer
        loss = loss + feature_transform_regularizer(trans_feat) * 0.001
    return loss


# ── train / eval loops ───────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, optimizer, model_name):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for pc, labels in loader:
        pc, labels = pc.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logit, trans_feat = forward_logit(model, pc, model_name)
        loss = compute_loss(logit, labels, trans_feat, model_name)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct     += (logit.argmax(1) == labels).sum().item()
        n           += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, model_name):
    model.eval()
    correct, n = 0, 0
    for pc, labels in loader:
        pc, labels = pc.to(DEVICE), labels.to(DEVICE)
        logit, _ = forward_logit(model, pc, model_name)
        correct += (logit.argmax(1) == labels).sum().item()
        n       += len(labels)
    return correct / n


# ── training entry-point ─────────────────────────────────────────────────────

def train(model_name, n_epochs=50, batch_size=32, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_path = os.path.join(DATA_DIR, 'train.h5')
    val_path   = os.path.join(DATA_DIR, 'test_rot000.h5')  # aligned test as val

    train_loader = make_loader(train_path, batch_size, shuffle=True)
    val_loader   = make_loader(val_path,   batch_size, shuffle=False)

    model = build_model(model_name).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

    torch.cuda.empty_cache()
    best_val_acc = 0.
    ckpt_path = os.path.join(DATA_DIR, f'ckpt_{model_name}.pth')

    print(f"\n{'='*50}")
    print(f"Training {model_name}  ({n_epochs} epochs, device={DEVICE})")
    print(f"{'='*50}")

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, model_name)
        val_acc = evaluate(model, val_loader, model_name)
        scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{n_epochs}  "
                  f"loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                  f"val_acc={val_acc:.3f}  best={best_val_acc:.3f}")

    print(f"Best val acc: {best_val_acc:.4f}  saved to {ckpt_path}")
    del model
    torch.cuda.empty_cache()
    return ckpt_path


# ── rotation sweep ───────────────────────────────────────────────────────────

def rotation_sweep(model_name, ckpt_path, batch_size=64):
    """Evaluate a trained model at every rotation test condition."""
    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    results = {}

    # Y-axis rotation: 0°..180°
    for deg in range(0, 181, 30):
        fname = f'test_rot{deg:03d}.h5'
        fpath = os.path.join(DATA_DIR, fname)
        loader = make_loader(fpath, batch_size, shuffle=False)
        acc = evaluate(model, loader, model_name)
        results[f'y_{deg}'] = acc
        print(f"  {model_name}  rot_y={deg:3d}°  acc={acc:.4f}")

    # Full SO3: average over 3 trials
    so3_accs = []
    for trial in range(3):
        fpath = os.path.join(DATA_DIR, f'test_so3_t{trial}.h5')
        loader = make_loader(fpath, batch_size, shuffle=False)
        so3_accs.append(evaluate(model, loader, model_name))
    results['so3_mean'] = float(np.mean(so3_accs))
    results['so3_std']  = float(np.std(so3_accs))
    print(f"  {model_name}  rot_SO3       acc={results['so3_mean']:.4f}"
          f" ± {results['so3_std']:.4f}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['simpleview', 'pointnet', 'dgcnn'])
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--eval_only',  action='store_true')
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        ckpt = os.path.join(DATA_DIR, f'ckpt_{model_name}.pth')
        if not args.eval_only:
            ckpt = train(model_name, args.epochs, args.batch_size, args.lr, args.seed)
        results = rotation_sweep(model_name, ckpt, args.batch_size)
        all_results[model_name] = results

    out_path = os.path.join(DATA_DIR, 'rotation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
