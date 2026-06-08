# analysis.py — diagnósticos, Grad-CAM e testes estatísticos.
# Contrato: geradores entregam [0,255] e o modelo normaliza internamente.
# Nada que monte tensor de inferência deve dividir por 255.

import json
import os
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


GRAPHS_DIR = 'outputs/graphs'

# Instância CLAHE compartilhada.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    return img


def _load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _apply_equalization(gray, mode):
    # 'hist' = equalização global; 'adaptive' = CLAHE; outro = sem mudança.
    if mode == 'hist':
        return cv2.equalizeHist(gray)
    if mode == 'adaptive':
        return _CLAHE.apply(gray)
    return gray


def analyze_contrast(df, graphs_dir=GRAPHS_DIR, sample_n=300, random_state=42):
    # Contraste (std de pixel) por imagem antes/depois de cada equalização, por classe.
    print("\n[Analysis 1] Contrast by class...")
    os.makedirs(graphs_dir, exist_ok=True)

    sample = (df[df['height'].notna()]
              .groupby('label', group_keys=False)
              .apply(lambda g: g.sample(min(sample_n, len(g)), random_state=random_state)))

    records = []
    for _, row in sample.iterrows():
        gray = _load_gray(row['path'])
        records.append({'label': row['label'], 'method': 'original',
                         'contrast': float(gray.std())})
        for mode in ('hist', 'adaptive'):
            eq = _apply_equalization(gray, mode)
            records.append({'label': row['label'], 'method': mode,
                             'contrast': float(eq.std())})

    results = pd.DataFrame(records)
    _plot_contrast_boxplot(results, graphs_dir)

    summary = (results.groupby(['label', 'method'])['contrast']
               .agg(['mean', 'std']).round(3).reset_index())
    summary.to_csv(f'{graphs_dir}/contrast_summary.csv', index=False)

    print(summary.to_string(index=False))
    return results


def _plot_contrast_boxplot(results, graphs_dir):
    methods = results['method'].unique()
    labels  = results['label'].unique()

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        data = [results[(results['method'] == method) &
                        (results['label']  == lbl)]['contrast'].values
                for lbl in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#4CAF50', '#F44336']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(method.capitalize())
        ax.set_xlabel('Class')
        if ax == axes[0]:
            ax.set_ylabel('Contrast (pixel std)')

    plt.suptitle('Contrast Distribution by Class and Method', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/contrast_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_intensity_distribution(df, graphs_dir=GRAPHS_DIR,
                                    sample_n=300, random_state=42):
    # Histograma médio de intensidade por classe, antes/depois de equalizar.
    print("\n[Analysis 2] Intensity distribution by class...")
    os.makedirs(graphs_dir, exist_ok=True)

    sample  = (df[df['height'].notna()]
               .groupby('label', group_keys=False)
               .apply(lambda g: g.sample(min(sample_n, len(g)), random_state=random_state)))
    methods = ['original', 'hist', 'adaptive']
    labels  = sample['label'].unique()
    bins    = np.linspace(0, 255, 64)

    histograms = {(m, lbl): np.zeros(len(bins) - 1) for m in methods for lbl in labels}
    counts     = {(m, lbl): 0                        for m in methods for lbl in labels}

    for _, row in sample.iterrows():
        gray = _load_gray(row['path'])
        lbl  = row['label']
        for mode in methods:
            img       = _apply_equalization(gray, mode) if mode != 'original' else gray
            h, _      = np.histogram(img.flatten(), bins=bins, density=True)
            histograms[(mode, lbl)] += h
            counts[(mode, lbl)]     += 1

    for key in histograms:
        histograms[key] /= counts[key]

    _plot_intensity_histograms(histograms, methods, labels, bins, graphs_dir)
    print("  Saved: intensity_histograms.png")


def _plot_intensity_histograms(histograms, methods, labels, bins, graphs_dir):
    colors      = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336'}
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for lbl in labels:
            ax.plot(bin_centers, histograms[(method, lbl)],
                    label=lbl, color=colors[lbl], linewidth=1.8)
        ax.set_title(method.capitalize())
        ax.set_xlabel('Pixel intensity')
        if ax == axes[0]:
            ax.set_ylabel('Density')
        ax.legend()

    plt.suptitle('Mean Intensity Distribution by Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/intensity_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_gradcam(model, image_paths, labels, target_size=(224, 224),
                graphs_dir=GRAPHS_DIR, n_samples=4):
    # Heatmaps Grad-CAM comparando original vs equalizado.
    # Imagens entram no modelo como float32 [0,255]; NÃO dividir por 255 aqui.
    print(f"\n[Analysis 3] Grad-CAM on {n_samples} samples...")
    os.makedirs(graphs_dir, exist_ok=True)

    grad_model, last_conv_name = _build_gradcam_model(model)
    print(f"  Using layer: {last_conv_name}")

    paths  = image_paths[:n_samples]
    lbls   = labels[:n_samples]
    modes  = ['original', 'hist', 'adaptive']
    n_rows = len(paths)

    fig, axes = plt.subplots(n_rows, len(modes), figsize=(5 * len(modes), 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, (path, lbl) in enumerate(zip(paths, lbls)):
        for col, mode in enumerate(modes):
            img_rgb   = _load_rgb(path)
            img_input = _prepare_gradcam_input(img_rgb, target_size, mode)
            heatmap   = _compute_gradcam(grad_model, img_input)
            overlay   = _overlay_heatmap(img_rgb, heatmap, target_size)

            axes[row][col].imshow(overlay)
            axes[row][col].set_title(f'{mode.capitalize()} | {lbl}', fontsize=9)
            axes[row][col].axis('off')

    plt.suptitle('Grad-CAM: Model Attention by Preprocessing Method', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/gradcam_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gradcam_comparison.png")


def _build_gradcam_model(model):
    # Retorna (grad_model, nome da última conv). grad_model devolve
    # (ativação da última conv, predição) num único forward.
    # As conv do backbone têm múltiplos inbound nodes (uso interno + uso no modelo
    # externo), então acessar layer.output direto levanta erro. Construímos
    # backbone_cam a partir de backbone.inputs (contexto de nó único) e recompomos
    # com a preprocess e a cabeça. model.layers vem em ordem topológica.

    backbone = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if backbone is None:
        raise ValueError("No backbone sub-model found in model.")

    last_conv = next(
        (l for l in reversed(backbone.layers) if isinstance(l, tf.keras.layers.Conv2D)),
        None
    )
    if last_conv is None:
        raise ValueError("No Conv2D layer found in backbone.")

    # backbone_cam: backbone.inputs -> [last_conv.output, backbone.output] (nó único).
    backbone_cam = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv.output, backbone.output],
        name='backbone_cam',
    )

    new_input    = tf.keras.Input(shape=model.input_shape[1:])
    preprocessed = model.get_layer('preprocess')(new_input)
    conv_out, bb_out = backbone_cam(preprocessed)

    # Aplica a cabeça em ordem topológica (GAP -> BN -> Dense -> ...), reusando pesos.
    x = bb_out
    past_backbone = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.name == 'preprocess':
            continue
        if isinstance(layer, tf.keras.Model):   # o backbone
            past_backbone = True
            continue
        if past_backbone:
            x = layer(x)   # reusa pesos; modo training propagado na chamada

    grad_model = tf.keras.Model(
        inputs=new_input,
        outputs=[conv_out, x],
        name='gradcam_model',
    )
    return grad_model, last_conv.name


def _prepare_gradcam_input(img_rgb, target_size, mode):
    # Redimensiona e (opcional) equaliza; retorna batch float32 [0,255] (sem rescale).
    if mode != 'original':
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = _apply_equalization(gray, mode)
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = img_rgb

    resized = cv2.resize(img, (target_size[1], target_size[0]))
    return tf.cast(np.expand_dims(resized, axis=0), tf.float32)


@tf.function
def _gradcam_forward_and_grad(grad_model, img_tensor):
    # Forward + d(loss)/d(conv_output). conv_output é saída explícita do grad_model,
    # então a tape registra o caminho até a predição sem watch explícito.
    # @tf.function mantém em grafo e reusa o trace (shape de img_tensor constante).
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, 0]
    return conv_output, tape.gradient(loss, conv_output)


def _compute_gradcam(grad_model, img_tensor):
    # Heatmap Grad-CAM normalizado para o batch dado.
    conv_output, grads = _gradcam_forward_and_grad(grad_model, img_tensor)

    if grads is None:
        raise RuntimeError(
            "_compute_gradcam: tape.gradient returned None. "
            "Verify that grad_model outputs the target conv layer tensor directly."
        )

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam     = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1).numpy()
    cam     = np.maximum(cam, 0)
    return cam / (cam.max() + 1e-8)


def _overlay_heatmap(img_rgb, heatmap, target_size):
    # Sobrepõe o heatmap na imagem (blend 60/40).
    h, w            = target_size
    heatmap_u8      = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(cv2.resize(heatmap_u8, (w, h)), cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_resized     = cv2.resize(img_rgb, (w, h))
    return np.uint8(img_resized * 0.6 + heatmap_rgb * 0.4)


def verify_datagen_preprocessing(pipeline, sample_df, target_size=(224, 224),
                                  graphs_dir=GRAPHS_DIR, n_images=4):
    # Pega um batch de cada datagen e compara estatísticas de pixel para confirmar
    # que preprocessing_function está sendo aplicada.
    from training import SEED

    os.makedirs(graphs_dir, exist_ok=True)
    modes = ['baseline', 'hist', 'adaptive']

    stats   = []
    batches = {}

    for mode in modes:
        datagen = pipeline.get_datagen_for_experiment(mode)
        gen     = datagen.flow_from_dataframe(
            dataframe   = sample_df[['path', 'label']].rename(
                              columns={'path': 'filename', 'label': 'class'}),
            x_col       = 'filename',
            y_col       = 'class',
            target_size = target_size,
            color_mode  = 'rgb',
            class_mode  = 'binary',
            batch_size  = len(sample_df),
            shuffle     = False,
        )
        imgs           = next(gen)[0]
        batches[mode]  = imgs

        stats.append({
            'mode': mode,
            'mean': float(imgs.mean()),
            'std' : float(imgs.std()),
            'min' : float(imgs.min()),
            'max' : float(imgs.max()),
        })
        print(f"  [{mode:>10}] mean={imgs.mean():.1f} | std={imgs.std():.1f} | "
              f"min={imgs.min():.1f} | max={imgs.max():.1f}  (range [0,255])")

    _check_identical_batches(batches)

    pd.DataFrame(stats).to_csv(f'{graphs_dir}/datagen_verification.csv', index=False)
    _plot_datagen_grid(batches, modes, n_images, graphs_dir)
    print("\n  Saved: datagen_verification.png / datagen_verification.csv")

    return pd.DataFrame(stats)


def _check_identical_batches(batches):
    # Avisa se dois modos produzem batches idênticos (preprocessing pode estar inativa).
    modes = list(batches.keys())
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            if np.allclose(batches[a], batches[b], atol=1e-6):
                print(f"\n  [WARNING] '{a}' and '{b}' produced identical batches — "
                      f"preprocessing_function may not be active for '{b}'.")
            else:
                max_diff = np.abs(batches[a] - batches[b]).max()
                print(f"  [OK] '{a}' vs '{b}' differ — max pixel diff: {max_diff:.1f}")


def _plot_datagen_grid(batches, modes, n_images, graphs_dir):
    # Grade: linhas = modos, colunas = imagens. Pixels p/ [0,1] só para exibição.
    n_rows, n_cols = len(modes), n_images
    fig, axes      = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for row, mode in enumerate(modes):
        imgs = batches[mode]
        for col in range(n_cols):
            img = np.clip(imgs[col] / 255.0, 0, 1)
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if col == 0:
                axes[row][col].set_ylabel(mode, fontsize=10, fontweight='bold')

    plt.suptitle('Datagen Output by Mode (same images, different preprocessing)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/datagen_verification.png', dpi=150, bbox_inches='tight')
    plt.close()


def _sig_level(p, alpha=0.05):
    # Faixas: *** p<=0.001 | ** p<=0.01 | * p<=alpha | ns caso contrário.
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= alpha:
        return '*'
    return 'ns'


def run_statistical_tests(logs_dir='outputs/logs', graphs_dir=GRAPHS_DIR,
                           metric='f1_macro', alpha=0.05):
    # Testes pareados não-paramétricos entre experimentos.
    # Test 1: arquiteturas (Context A). Test 2: preprocessing (Context B).
    # Test 3: augmentation (Context A, Wilcoxon).
    from scipy.stats import friedmanchisquare, wilcoxon

    os.makedirs(graphs_dir, exist_ok=True)
    report = []

    def load_cv(experiment):
        path = os.path.join(logs_dir, experiment, 'cv_metrics.csv')
        if not os.path.exists(path):
            print(f"  [SKIP] {experiment} — cv_metrics.csv not found.")
            return None
        df = pd.read_csv(path)
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not in {path}.")
        return df

    def pivot_by_arch(df):
        return {
            arch: df[df['arch'] == arch].sort_values('fold')[metric].values
            for arch in df['arch'].unique()
        }

    # Test 1
    print("\n" + "="*60)
    print("  TEST 1 — Architecture comparison (Context A baseline)")
    print("="*60)

    df_base = load_cv('baseline')

    if df_base is not None:
        arch_scores = pivot_by_arch(df_base)
        archs       = list(arch_scores.keys())

        if len(archs) >= 3:
            stat, p = friedmanchisquare(*[arch_scores[a] for a in archs])
            sig     = p <= alpha
            level   = _sig_level(p, alpha)
            print(f"\n  Friedman | stat={stat:.4f} | p={p:.4f} | "
                  f"{'significant' if sig else 'not significant'} [{level}] (alpha={alpha})")

            report.append({
                'test': 'Friedman', 'context': 'architecture_comparison',
                'group_a': ' vs '.join(archs), 'group_b': '',
                'stat': round(stat, 4), 'p_value': round(p, 6),
                'significant': sig, 'sig_level': level,
                'note': 'H0: no difference between architectures',
            })

            if sig:
                print("\n  Post-hoc Nemenyi:")
                nemenyi = _nemenyi_test(arch_scores, archs)
                for (a, b), p_pair in nemenyi.items():
                    sig_pair   = p_pair <= alpha
                    level_pair = _sig_level(p_pair, alpha)
                    print(f"    {a} vs {b} | p={p_pair:.4f} | {level_pair}")
                    report.append({
                        'test': 'Nemenyi', 'context': 'architecture_comparison',
                        'group_a': a, 'group_b': b, 'stat': '',
                        'p_value': round(p_pair, 6), 'significant': sig_pair,
                        'sig_level': level_pair, 'note': 'post-hoc after Friedman',
                    })
                _plot_critical_difference(
                    arch_scores, archs, alpha,
                    title='Architecture Comparison — Context A',
                    filename='cd_architecture', graphs_dir=graphs_dir,
                )

    # Test 2
    print("\n" + "="*60)
    print("  TEST 2 — Preprocessing effect (Context B)")
    print("="*60)

    context_b = ['baseline_sample', 'hist_sample', 'adaptive_sample']
    dfs_b     = {exp: load_cv(exp) for exp in context_b}

    if all(v is not None for v in dfs_b.values()):
        for arch in dfs_b['baseline_sample']['arch'].unique():
            scores = {
                exp: dfs_b[exp][dfs_b[exp]['arch'] == arch]
                         .sort_values('fold')[metric].values
                for exp in context_b
            }

            if any(len(v) < 2 for v in scores.values()):
                print(f"\n  [{arch}] Not enough data — skipping.")
                continue

            stat, p = friedmanchisquare(*[scores[e] for e in context_b])
            sig     = p <= alpha
            level   = _sig_level(p, alpha)
            print(f"\n  [{arch}] Friedman | stat={stat:.4f} | p={p:.4f} | "
                  f"{'significant' if sig else 'not significant'} [{level}]")

            report.append({
                'test': 'Friedman', 'context': f'preprocessing_{arch}',
                'group_a': ' vs '.join(context_b), 'group_b': '',
                'stat': round(stat, 4), 'p_value': round(p, 6),
                'significant': sig, 'sig_level': level,
                'note': 'H0: no difference between preprocessing methods',
            })

            if sig:
                print(f"  [{arch}] Post-hoc Nemenyi:")
                nemenyi = _nemenyi_test(scores, context_b)
                for (a, b), p_pair in nemenyi.items():
                    sig_pair   = p_pair <= alpha
                    level_pair = _sig_level(p_pair, alpha)
                    print(f"    {a} vs {b} | p={p_pair:.4f} | {level_pair}")
                    report.append({
                        'test': 'Nemenyi', 'context': f'preprocessing_{arch}',
                        'group_a': a, 'group_b': b, 'stat': '',
                        'p_value': round(p_pair, 6), 'significant': sig_pair,
                        'sig_level': level_pair, 'note': 'post-hoc after Friedman',
                    })

    # Test 3
    print("\n" + "="*60)
    print("  TEST 3 — Augmentation effect (Context A, Wilcoxon)")
    print("="*60)

    df_aug = load_cv('augmented')
    if df_base is not None and df_aug is not None:
        base_scores = pivot_by_arch(df_base)
        aug_scores  = pivot_by_arch(df_aug)

        for arch in [a for a in base_scores if a in aug_scores]:
            a_scores = base_scores[arch]
            b_scores = aug_scores[arch]

            if np.allclose(a_scores, b_scores):
                print(f"\n  [{arch}] Scores identical — Wilcoxon not applicable.")
                continue

            stat, p = wilcoxon(a_scores, b_scores, alternative='two-sided')
            sig     = p <= alpha
            level   = _sig_level(p, alpha)
            print(f"\n  [{arch}] Wilcoxon | stat={stat:.4f} | p={p:.4f} | "
                  f"{'significant' if sig else 'not significant'} [{level}]")

            report.append({
                'test': 'Wilcoxon', 'context': f'augmentation_{arch}',
                'group_a': 'baseline', 'group_b': 'augmented',
                'stat': round(stat, 4), 'p_value': round(p, 6),
                'significant': sig, 'sig_level': level,
                'note': 'H0: no difference between baseline and augmented',
            })

    df_report = pd.DataFrame(report)
    save_path = os.path.join(graphs_dir, 'statistical_report.csv')
    df_report.to_csv(save_path, index=False)
    print(f"\n  Report saved to: {save_path}")

    _print_summary(df_report, alpha)
    return df_report


def _nemenyi_test(scores_dict, groups):
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("  [WARNING] scikit-posthocs not installed. Run: pip install scikit-posthocs")
        return {}

    data   = np.array([scores_dict[g] for g in groups]).T
    matrix = sp.posthoc_nemenyi_friedman(pd.DataFrame(data, columns=groups))
    return {(a, b): float(matrix.loc[a, b]) for a, b in combinations(groups, 2)}


def _plot_critical_difference(scores_dict, groups, alpha, title, filename, graphs_dir):
    rank_data  = np.array([scores_dict[g] for g in groups]).T
    ranks      = np.array([_rank_row(row) for row in rank_data])
    mean_ranks = ranks.mean(axis=0)

    sorted_idx    = np.argsort(mean_ranks)
    sorted_groups = [groups[i] for i in sorted_idx]
    sorted_ranks  = mean_ranks[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(sorted_ranks, [1] * len(sorted_groups), s=80, zorder=3)

    for i, (g, r) in enumerate(zip(sorted_groups, sorted_ranks)):
        ax.annotate(g, (r, 1), textcoords='offset points',
                    xytext=(0, 10 + (i % 2) * 14), ha='center', fontsize=9)

    ax.set_xlim(0.5, len(groups) + 0.5)
    ax.set_xlabel('Mean rank (lower = better)')
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()


def _rank_row(row):
    return np.argsort(np.argsort(-row)).astype(float) + 1


def _print_summary(df_report, alpha):
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    for is_sig, label in [(True, f"Significant (p <= {alpha})"),
                           (False, f"Non-significant (p > {alpha})")]:
        subset = df_report[df_report['significant'] == is_sig]
        if subset.empty:
            continue
        print(f"\n  {label}:")
        for _, row in subset.iterrows():
            group = row['group_a'] if not row['group_b'] else f"{row['group_a']} vs {row['group_b']}"
            print(f"    [{row['test']}] {row['context']} | {group} | "
                  f"p={row['p_value']} {row['sig_level']}")


def reextract_test_metrics(experiment, pipeline, val_datagen,
                            logs_dir='outputs/logs', models_dir='outputs/models'):
    # Reconstrói test_metrics.csv a partir dos checkpoints da fase 2.
    from training import ARCHITECTURES, evaluate_test_set, load_trained_model

    exp_logs   = os.path.join(logs_dir,   experiment)
    exp_models = os.path.join(models_dir, experiment)
    cv_path    = os.path.join(exp_logs, 'cv_metrics.csv')

    if not os.path.exists(cv_path):
        raise FileNotFoundError(
            f"cv_metrics.csv not found at '{cv_path}'. "
            "Cannot determine best fold without cross-validation results."
        )

    cv_df        = pd.read_csv(cv_path)
    test_metrics = []

    for arch in ARCHITECTURES:
        arch_cv = cv_df[cv_df['arch'] == arch]
        if arch_cv.empty:
            print(f"  [{experiment}] {arch} — not in cv_metrics, skipping.")
            continue

        best_fold = int(arch_cv.loc[arch_cv['f1_macro'].idxmax(), 'fold'])
        ckpt_path = os.path.join(exp_models, f'{arch}_fold{best_fold}_phase2_best')

        if not os.path.exists(ckpt_path + '.index'):
            print(f"  [{experiment}] {arch} fold {best_fold} — "
                  f"checkpoint not found at '{ckpt_path}', skipping.")
            continue

        print(f"\n  [{experiment}] {arch} — best fold: {best_fold} | "
              f"f1_macro: {arch_cv['f1_macro'].max():.4f}")

        model   = load_trained_model(arch, ckpt_path)
        metrics = evaluate_test_set(model, pipeline.X_test, pipeline.y_test,
                                    val_datagen, arch)
        test_metrics.append(metrics)

    if not test_metrics:
        print(f"  [{experiment}] No metrics extracted — check checkpoints.")
        return None

    df = pd.DataFrame(test_metrics)
    df.to_csv(os.path.join(exp_logs, 'test_metrics.csv'), index=False)

    json_path = os.path.join(exp_logs, 'test_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n  [{experiment}] test_metrics saved to '{exp_logs}/'")
    return df


def reextract_cv_metrics(experiment, pipeline, val_datagen,
                          logs_dir='outputs/logs', models_dir='outputs/models'):
    # Reconstrói cv_metrics.csv a partir dos checkpoints da fase 2.
    from training import (ARCHITECTURES, compute_metrics, load_trained_model,
                          make_generator, predict)

    exp_logs   = os.path.join(logs_dir,   experiment)
    exp_models = os.path.join(models_dir, experiment)
    os.makedirs(exp_logs, exist_ok=True)

    cv_metrics = []

    for arch in ARCHITECTURES:
        for fold_data in pipeline.folds:
            fold      = fold_data['fold']
            ckpt_path = os.path.join(exp_models, f'{arch}_fold{fold}_phase2_best')

            if not os.path.exists(ckpt_path + '.index'):
                print(f"  [{experiment}] {arch} fold {fold} — checkpoint not found, skipping.")
                continue

            print(f"  [{experiment}] {arch} fold {fold} — loading {ckpt_path}")

            model          = load_trained_model(arch, ckpt_path)
            val_gen        = make_generator(val_datagen, fold_data['X_val'],
                                            fold_data['y_val'], shuffle=False)
            y_true, y_pred = predict(model, val_gen, fold_data['y_val'])
            metrics        = compute_metrics(y_true, y_pred, fold, arch)
            cv_metrics.append(metrics)

            print(f"    f1_macro={metrics['f1_macro']:.4f} | "
                  f"recall_macro={metrics['recall_macro']:.4f}")

    if not cv_metrics:
        print(f"  [{experiment}] No metrics extracted — check checkpoints.")
        return None

    df = pd.DataFrame(cv_metrics)
    df.to_csv(os.path.join(exp_logs, 'cv_metrics.csv'), index=False)
    print(f"\n  [{experiment}] cv_metrics saved to '{exp_logs}/'")
    return df
