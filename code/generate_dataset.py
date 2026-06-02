"""
Control dataset generator: 5 synthetic 3D shapes with controlled rotation.

Hypothesis tested (van Gemert RM9, RP7, RP9):
  "SimpleView degrades more gracefully under 3D rotation than point-based
   methods (PointNet, DGCNN) when trained without rotation augmentation."

Single controlled variable (RP9): rotation angle theta applied at test time.
All other variables are held constant: shape geometry, N_POINTS, training
protocol, loss function, model selection strategy.

Classes: cube, sphere, cylinder, cone, torus  (5 classes, clearly distinct)
Train:  400 samples / class, canonical orientation (no rotation)
Test:   100 samples / class per rotation condition
"""

import numpy as np
import h5py
import os
from scipy.spatial.transform import Rotation

SEED = 42
N_POINTS = 1024
N_TRAIN_PER_CLASS = 400
N_TEST_PER_CLASS = 100
CLASS_NAMES = ['cube', 'sphere', 'cylinder', 'cone', 'torus']
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_shapes')


# ── shape samplers ──────────────────────────────────────────────────────────

def sample_cube(n):
    face_n = n // 6
    extra = n - face_n * 6
    pts = []
    for i in range(6):
        k = face_n + (1 if i < extra else 0)
        u = np.random.uniform(-1, 1, (k, 2))
        if   i == 0: face = np.c_[u[:, 0], u[:, 1],  np.ones(k)]
        elif i == 1: face = np.c_[u[:, 0], u[:, 1], -np.ones(k)]
        elif i == 2: face = np.c_[u[:, 0],  np.ones(k), u[:, 1]]
        elif i == 3: face = np.c_[u[:, 0], -np.ones(k), u[:, 1]]
        elif i == 4: face = np.c_[ np.ones(k), u[:, 0], u[:, 1]]
        else:        face = np.c_[-np.ones(k), u[:, 0], u[:, 1]]
        pts.append(face)
    return np.vstack(pts)


def sample_sphere(n):
    pts = np.random.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def sample_cylinder(n):
    # lateral area : 2*pi*r*h = 4*pi (r=1, h=2)
    # each cap     : pi*r^2   = pi
    total = 6 * np.pi
    n_lat  = int(n * 4 * np.pi / total)
    n_top  = int(n * np.pi / total)
    n_bot  = n - n_lat - n_top

    theta = np.random.uniform(0, 2 * np.pi, n_lat)
    z     = np.random.uniform(-1, 1, n_lat)
    lat   = np.c_[np.cos(theta), np.sin(theta), z]

    r = np.sqrt(np.random.uniform(0, 1, n_top))
    t = np.random.uniform(0, 2 * np.pi, n_top)
    top = np.c_[r * np.cos(t), r * np.sin(t), np.ones(n_top)]

    r = np.sqrt(np.random.uniform(0, 1, n_bot))
    t = np.random.uniform(0, 2 * np.pi, n_bot)
    bot = np.c_[r * np.cos(t), r * np.sin(t), -np.ones(n_bot)]

    return np.vstack([lat, top, bot])


def sample_cone(n):
    # apex at z=+1, base radius=1 at z=-1, height=2
    # lateral slant = sqrt(1^2 + 2^2) = sqrt(5)
    # area_lat = pi * r * slant = pi * sqrt(5)
    # area_base = pi
    slant = np.sqrt(5)
    frac_lat = slant / (slant + 1)
    n_lat  = int(n * frac_lat)
    n_base = n - n_lat

    # uniform lateral sampling: r proportional to distance from apex
    u     = np.random.uniform(0, 1, n_lat)
    r_lat = np.sqrt(u)           # area-correct sampling
    theta = np.random.uniform(0, 2 * np.pi, n_lat)
    z_lat = 1 - 2 * r_lat        # z from +1 (apex) to -1 (base)
    lat   = np.c_[r_lat * np.cos(theta), r_lat * np.sin(theta), z_lat]

    r = np.sqrt(np.random.uniform(0, 1, n_base))
    t = np.random.uniform(0, 2 * np.pi, n_base)
    base = np.c_[r * np.cos(t), r * np.sin(t), -np.ones(n_base)]

    return np.vstack([lat, base])


def sample_torus(n, R=0.65, r=0.25):
    """Rejection-sample uniform torus surface."""
    pts = []
    needed = n
    while needed > 0:
        batch = max(needed * 4, 8192)
        theta = np.random.uniform(0, 2 * np.pi, batch)
        phi   = np.random.uniform(0, 2 * np.pi, batch)
        accept = np.random.uniform(0, 1, batch) < (R + r * np.cos(theta)) / (R + r)
        theta, phi = theta[accept], phi[accept]
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        pts.append(np.c_[x, y, z])
        needed -= accept.sum()
    return np.vstack(pts)[:n]


SAMPLERS = [sample_cube, sample_sphere, sample_cylinder, sample_cone, sample_torus]


# ── normalisation ────────────────────────────────────────────────────────────

def normalise(pts):
    """Centre at origin, scale to fit in unit sphere."""
    pts = pts - pts.mean(axis=0)
    pts = pts / np.linalg.norm(pts, axis=1).max()
    return pts.astype(np.float32)


# ── rotations ────────────────────────────────────────────────────────────────

def rot_y(pts, deg):
    a  = np.deg2rad(deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)],
                   [0,         1, 0         ],
                   [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)
    return pts @ Ry.T


def rot_so3(pts):
    R = Rotation.random().as_matrix().astype(np.float32)
    return pts @ R.T


# ── split generation ─────────────────────────────────────────────────────────

def generate_split(n_per_class, rot_type='none', rot_deg=0):
    """
    rot_type: 'none' | 'y' | 'so3'
    rot_deg:  scalar (only used when rot_type == 'y')
    """
    pcs, labels = [], []
    for cls, sampler in enumerate(SAMPLERS):
        for _ in range(n_per_class):
            pts = normalise(sampler(N_POINTS))
            if   rot_type == 'y':   pts = rot_y(pts, rot_deg)
            elif rot_type == 'so3': pts = rot_so3(pts)
            pcs.append(pts)
            labels.append(cls)
    pcs    = np.stack(pcs)
    labels = np.array(labels, dtype=np.int64)
    idx    = np.random.permutation(len(pcs))
    return pcs[idx], labels[idx]


def save_h5(path, pcs, labels):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data',  data=pcs)
        f.create_dataset('label', data=labels.reshape(-1, 1))
    print(f"  saved {path}  shape={pcs.shape}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating dataset in {OUT_DIR}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train: {N_TRAIN_PER_CLASS} / class, Test: {N_TEST_PER_CLASS} / class")

    # Training set: canonical orientation (no rotation)
    pcs, labels = generate_split(N_TRAIN_PER_CLASS, rot_type='none')
    save_h5(os.path.join(OUT_DIR, 'train.h5'), pcs, labels)

    # Test: aligned (no rotation) — verifies c1, c2
    pcs, labels = generate_split(N_TEST_PER_CLASS, rot_type='none')
    save_h5(os.path.join(OUT_DIR, 'test_rot000.h5'), pcs, labels)

    # Test: Y-axis rotation sweep 30°..180° — isolates rotation as single variable
    for deg in range(30, 181, 30):
        pcs, labels = generate_split(N_TEST_PER_CLASS, rot_type='y', rot_deg=deg)
        save_h5(os.path.join(OUT_DIR, f'test_rot{deg:03d}.h5'), pcs, labels)

    # Test: full SO3 random rotation (3 seeds for stability)
    for trial in range(3):
        np.random.seed(SEED + trial + 1)
        pcs, labels = generate_split(N_TEST_PER_CLASS, rot_type='so3')
        save_h5(os.path.join(OUT_DIR, f'test_so3_t{trial}.h5'), pcs, labels)

    print("Done.")
