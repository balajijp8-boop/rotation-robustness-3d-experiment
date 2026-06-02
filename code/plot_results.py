"""
Plot accuracy vs rotation angle for all three models.
Produces the main figure for the blog post.
Per van Gemert RP11: script all figures.
"""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_shapes')
RESULTS_JSON = os.path.join(DATA_DIR, 'rotation_results.json')
OUT_FIG    = os.path.join(DATA_DIR, 'rotation_accuracy.pdf')

COLORS = {'simpleview': '#1f77b4', 'pointnet': '#d62728', 'dgcnn': '#2ca02c'}
LABELS = {'simpleview': 'SimpleView (projection-based)',
          'pointnet':   'PointNet (point-based)',
          'dgcnn':      'DGCNN (point-based)'}
DEGREES = list(range(0, 181, 30))


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        'Accuracy vs. rotation angle\n'
        '(trained on aligned data, no rotation augmentation)',
        fontsize=12, fontweight='bold')

    # ── left: Y-axis rotation sweep ─────────────────────────────────────────
    ax = axes[0]
    for name, res in results.items():
        accs = [res[f'y_{d}'] * 100 for d in DEGREES]
        ax.plot(DEGREES, accs, marker='o', color=COLORS[name],
                label=LABELS[name], linewidth=2, markersize=6)

    ax.set_xlabel('Y-axis rotation angle (degrees)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Y-axis rotation sweep\n(single confounder, RP9)', fontsize=10)
    ax.set_xticks(DEGREES)
    ax.set_ylim(0, 105)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotate the "problem" region
    ax.axvspan(90, 180, alpha=0.07, color='red')
    ax.text(130, 5, 'problem\nregion', ha='center', fontsize=8, color='red')

    # ── right: SO3 bar chart ──────────────────────────────────────────────
    ax = axes[1]
    names  = list(results.keys())
    means  = [results[n]['so3_mean'] * 100 for n in names]
    stds   = [results[n]['so3_std']  * 100 for n in names]
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6,
                  error_kw={'linewidth': 2}, alpha=0.85)
    ax.axhline(100 / 5, color='grey', linestyle='--', linewidth=1,
               label='random chance (20%)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n].split(' (')[0] for n in names], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Full SO3 random rotation\n(mean ± std over 3 seeds)', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches='tight')
    print(f"Figure saved to {OUT_FIG}")
    plt.savefig(OUT_FIG.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Figure saved to {OUT_FIG.replace('.pdf', '.png')}")


if __name__ == '__main__':
    if not os.path.exists(RESULTS_JSON):
        print(f"ERROR: {RESULTS_JSON} not found. Run train_eval.py first.")
        sys.exit(1)
    results = load_results()
    print("Loaded results for:", list(results.keys()))
    plot(results)
