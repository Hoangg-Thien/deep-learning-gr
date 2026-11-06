"""Noninteractive scientific plots with explicit split labels and raw units."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import gaussian_2d
from .training import predict

plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'axes.grid': True,
                     'grid.alpha': 0.25, 'savefig.bbox': 'tight'})


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grid():
    xx, yy = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-4, 4, 81))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def data_figures(splits, output):
    output = Path(output)
    xx, yy, xy = grid()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, gaussian_2d(xy).reshape(xx.shape), cmap='viridis')
    ax.set(xlabel='x1', ylabel='x2', zlabel='Density f(x)', title='Official Gaussian ground truth')
    save(fig, output / 'gaussian_surface.png')
    fig = plt.figure(figsize=(15, 4.5))
    for i, (name, (x, y)) in enumerate(splits.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y[:, 0], c=y[:, 0], cmap='viridis', s=8, alpha=0.7)
        ax.set(xlabel='x1', ylabel='x2', zlabel='Density', title=f'{name} (N={len(x)})')
    save(fig, output / 'data_scatter.png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, (x, y) in splits.items():
        for ax, values, label in zip(axes, [x[:, 0], x[:, 1], y[:, 0]], ['x1', 'x2', 'y']):
            ax.hist(values, bins=25, density=True, histtype='step', label=name)
            ax.set(xlabel=label, ylabel='Empirical density')
            ax.legend()
    save(fig, output / 'data_distributions.png')


def loss_figure(history, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [('train_loss', 'Train'), ('validation_loss', 'Validation')]:
        ax.semilogy([r['epoch'] for r in history], [r[key] for r in history], label=label)
    ax.set(xlabel='Epoch', ylabel='MSE (density squared)', title=title)
    ax.legend()
    save(fig, path)


def comparison_figures(summary, records, histories, output):
    output = Path(output)
    xlabels = {'A': 'Training samples A', 'B': 'Hidden neurons B (one layer)', 'C': 'Hidden layers C'}
    for group in 'ABC':
        rows = sorted([r for r in summary if r['experiment'] == group], key=lambda r: r['value'])
        if not rows:
            continue
        values = [r['value'] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ['mse', 'mae', 'rmse']):
            for split in ['train', 'validation', 'test']:
                ax.errorbar(values, [r[f'{split}_{metric}_mean'] for r in rows],
                            yerr=[r[f'{split}_{metric}_std'] for r in rows],
                            label=split, marker='o', capsize=3, markersize=3)
            ax.set(xlabel=xlabels[group], ylabel=metric.upper(), yscale='log', title=f'{group}: mean +/- sample SD')
            ax.legend()
        save(fig, output / f'{group}_comparison.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for value in values:
            subset = [r for r in records if r['experiment'] == group and r['value'] == value]
            # Different early-stop lengths: draw each seed, label first only.
            for i, record in enumerate(subset):
                history = histories[record['run_id']]
                color = plt.get_cmap('tab10')(values.index(value) % 10)
                for ax, key in zip(axes, ['train_loss', 'validation_loss']):
                    ax.semilogy([r['epoch'] for r in history], [r[key] for r in history],
                                color=color, alpha=0.65, label=str(value) if i == 0 else None)
        for ax, title in zip(axes, ['Train', 'Validation']):
            ax.set(xlabel='Epoch', ylabel='MSE', title=f'{group}: {title} convergence, all seeds')
            ax.legend(ncol=2, fontsize=8, title='A/B/C value')
        save(fig, output / f'{group}_convergence.png')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(values, [r['training_seconds_mean'] for r in rows],
                    yerr=[r['training_seconds_std'] for r in rows], marker='o', capsize=4)
        ax.set(xlabel=xlabels[group], ylabel='Training + validation time (seconds)', title=f'{group}: wall-clock time')
        save(fig, output / f'{group}_training_time.png')


def evaluation_figures(model, x, y, prediction, output):
    output = Path(output)
    xx, yy, xy = grid()
    truth = gaussian_2d(xy).reshape(xx.shape)
    estimate = predict(model, xy).reshape(xx.shape)
    fig = plt.figure(figsize=(16, 5))
    for i, (z, title, cmap) in enumerate([(truth, 'Ground truth', 'viridis'),
                                         (estimate, 'Selected MLP prediction', 'viridis'),
                                         (np.abs(estimate - truth), 'Absolute error on diagnostic grid', 'magma')], 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.plot_surface(xx, yy, z, cmap=cmap)
        ax.set(xlabel='x1', ylabel='x2', zlabel='Density / absolute density error', title=title)
    save(fig, output / 'selected_surfaces.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, prediction, s=25, alpha=0.7)
    limits = [min(0, float(prediction.min())), max(float(y.max()), float(prediction.max()))]
    axes[0].plot(limits, limits, '--', color='black', label='Ideal y = prediction')
    axes[0].set(xlabel='Ground truth density y', ylabel='Prediction', title='Independent test: 100 samples')
    axes[0].legend()
    sc = axes[1].scatter(x[:, 0], x[:, 1], c=(prediction - y).ravel(), cmap='coolwarm')
    axes[1].set(xlabel='x1', ylabel='x2', title='Test residual (prediction - y)')
    fig.colorbar(sc, ax=axes[1], label='Density error')
    save(fig, output / 'selected_test_predictions.png')
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')
    sizes = [2] + model.hidden_sizes + [1]
    for i, width in enumerate(sizes):
        ax.text(i, 0.5, f'{"Input" if i == 0 else "Output" if i == len(sizes)-1 else "Hidden"}\n{width} neurons',
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#e9f0f6', edgecolor='#345'))
        if i < len(sizes) - 1:
            ax.annotate('', (i + 0.77, 0.5), (i + 0.23, 0.5), arrowprops=dict(arrowstyle='->'))
    ax.set(xlim=(-0.5, len(sizes) - 0.5), ylim=(0, 1), title=f'MLP: {model.activation} hidden activations; linear output; bias at each Linear')
    save(fig, output / 'selected_architecture.png')
