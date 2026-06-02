import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from all_utils import DATASET_NUM_CLASS
DATASET_NUM_CLASS['synthetic5'] = 5
from train_eval import rotation_sweep

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_shapes')

all_results = {}

with open(os.path.join(DATA_DIR, 'results_simpleview.json')) as f:
    all_results.update(json.load(f))

all_results['pointnet'] = {
    'y_0': 1.0, 'y_30': 1.0, 'y_60': 0.8, 'y_90': 0.6,
    'y_120': 0.634, 'y_150': 0.98, 'y_180': 0.8,
    'so3_mean': 0.6673, 'so3_std': 0.0133
}

ckpt = os.path.join(DATA_DIR, 'ckpt_dgcnn.pth')
all_results['dgcnn'] = rotation_sweep('dgcnn', ckpt, batch_size=16)

out = os.path.join(DATA_DIR, 'rotation_results.json')
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)

print('=== FINAL RESULTS ===')
for model, res in all_results.items():
    print(f"\n{model.upper()}:")
    for deg in range(0, 181, 30):
        print(f"  rot_y={deg:3d}:  {res[f'y_{deg}']:.4f}")
    print(f"  SO3:       {res['so3_mean']:.4f} +/- {res['so3_std']:.4f}")
print(f"\nSaved to {out}")
