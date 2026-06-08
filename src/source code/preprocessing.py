# preprocessing.py — DataPipeline: load, EDA, clean, balance, split, datagens.
# Contrato: datagens entregam [0,255] sem rescale; o modelo normaliza internamente.

import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPipeline:

    CLASSES = {'NORMAL', 'PNEUMONIA'}
    COLORS  = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336'}

    # Instância CLAHE compartilhada — evita recriar a cada imagem.
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(
        self,
        base_path='datasets',
        target_size=(224, 224),
        graphs_dir='graphs',
        random_state=42,
        chestxray8_normal_cap=5000,
        n_splits=5,
        test_size=0.15,
        skip_plots=False,
    ):
        self.base_path             = base_path
        self.target_size           = target_size
        self.graphs_dir            = graphs_dir
        self.random_state          = random_state
        self.chestxray8_normal_cap = chestxray8_normal_cap
        self.n_splits              = n_splits
        self.test_size             = test_size
        self.skip_plots            = skip_plots

        self.df      = None
        self.folds   = []
        self.X_test  = None
        self.y_test  = None

        self._sample_path = None  # imagem usada nos plots de exemplo

        os.makedirs(self.graphs_dir, exist_ok=True)

    def run(self):
        # skip_plots=True (debug) pula só os plots; load/clean/balance/split rodam normal.
        self._load_metadata()
        self._print_overview()

        if not self.skip_plots:
            self._plot_class_distribution(suffix='_original')
            self._plot_dimension_distribution()
            self._plot_visual_samples()

        self._remove_corrupted()
        self._cap_chestxray8_normal()

        if not self.skip_plots:
            self._plot_class_distribution(suffix='_balanced')

        self._sample_path = self.df['path'].iloc[0]

        if not self.skip_plots:
            self._plot_preprocessing_example()
            self._plot_equalization_example()
            self._plot_augmentation_examples()

        self._split_kfold()
        self._save_splits_csv()

    @staticmethod
    def _safe_filename(name):
        for char in (' ', '-', '(', ')', '[', ']'):
            name = name.replace(char, '_')
        return name

    @staticmethod
    def _normalize_class(dir_name):
        upper = dir_name.upper()
        if 'PNEUMONIA' in upper:
            return 'PNEUMONIA'
        if 'NORMAL' in upper:
            return 'NORMAL'
        return None

    def _load_metadata(self):
        # Lê dimensões na carga para detectar arquivos corrompidos aqui, não no treino.
        print("Loading metadata...")
        records = []

        for dataset in os.listdir(self.base_path):
            dataset_path = os.path.join(self.base_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for dirpath, _, filenames in os.walk(dataset_path):
                label = self._normalize_class(os.path.basename(dirpath))
                if label is None:
                    continue

                for img_name in filenames:
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    path = os.path.join(dirpath, img_name)
                    img  = cv2.imread(path)

                    if img is None:
                        height = width = channels = None
                    else:
                        height, width = img.shape[:2]
                        channels      = img.shape[2] if img.ndim == 3 else 1

                    records.append({
                        'dataset' : dataset,
                        'label'   : label,
                        'image'   : img_name,
                        'path'    : path,
                        'height'  : height,
                        'width'   : width,
                        'channels': channels,
                    })

        self.df = pd.DataFrame(records)
        print(f"Total loaded: {len(self.df)} images")
        print(self.df.groupby(['dataset', 'label']).size()
                     .reset_index(name='count').to_string(index=False))

    def _print_overview(self):
        total = self.df.groupby('dataset').size().reset_index(name='total')
        dist  = self.df['label'].value_counts().reset_index()
        dist.columns = ['label', 'count']

        print("\nTotal per dataset:")
        print(total.to_string(index=False))
        print("\nTotal per class:")
        print(dist.to_string(index=False))
        print(f"\nTotal    : {len(self.df)}")
        print(f"Corrupted: {self.df['height'].isna().sum()}")

    def _plot_class_distribution(self, suffix=''):
        print(f"\nPlotting class distribution{suffix}...")
        datasets = self.df['dataset'].unique()
        n        = len(datasets)

        fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

        for ax, ds in zip(axes[:n], datasets):
            self._barplot(ax, self.df[self.df['dataset'] == ds]['label'].value_counts(), ds)
        self._barplot(axes[n], self.df['label'].value_counts(), 'UNIFIED SET')

        title = 'Class Distribution per Dataset + Unified Set'
        if suffix:
            title += f' ({suffix.strip("_").capitalize()})'

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/class_distribution{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _barplot(self, ax, counts, title):
        bars  = ax.bar(counts.index, counts.values,
                       color=[self.COLORS[c] for c in counts.index],
                       edgecolor='black', linewidth=0.7)
        total = counts.sum()
        pct   = (counts / total * 100).round(1)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=10)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f'{c}\n({pct[c]}%)' for c in counts.index])

    def _plot_dimension_distribution(self):
        print("\nPlotting dimension distribution...")
        datasets  = self.df['dataset'].unique()
        fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4 * len(datasets)))

        for i, ds in enumerate(datasets):
            df_ds = self.df[self.df['dataset'] == ds].dropna(subset=['height', 'width'])
            ax_h  = axes[i][0] if len(datasets) > 1 else axes[0]
            ax_w  = axes[i][1] if len(datasets) > 1 else axes[1]

            for ax, col, color, label in [
                (ax_h, 'height', 'steelblue',  'Height'),
                (ax_w, 'width',  'darkorange', 'Width'),
            ]:
                ax.hist(df_ds[col], bins=30, color=color, edgecolor='black', alpha=0.8)
                ax.set_title(f'[{ds}] {label}')
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Frequency')
                ax.axvline(df_ds[col].mean(), color='red', linestyle='--',
                           label=f"Mean: {df_ds[col].mean():.0f}px")
                ax.legend()

            print(f"\n  {ds}")
            for col in ['height', 'width']:
                s = df_ds[col]
                print(f"   {col.capitalize():7} — "
                      f"min: {s.min():.0f} | max: {s.max():.0f} | "
                      f"mean: {s.mean():.1f} | median: {s.median():.0f}")

        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/dimension_distribution.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_visual_samples(self):
        print("\nGenerating visual samples...")
        for ds in self.df['dataset'].unique():
            classes        = self.df[self.df['dataset'] == ds]['label'].unique()
            n_cols, n_rows = 4, len(classes)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            if n_rows == 1:
                axes = [axes]

            for row_idx, label in enumerate(classes):
                samples = self.df[
                    (self.df['dataset'] == ds) &
                    (self.df['label']   == label) &
                    (self.df['height'].notna())
                ].sample(min(n_cols, 4), random_state=self.random_state)

                for col_idx, (_, s) in enumerate(samples.iterrows()):
                    img = cv2.cvtColor(cv2.imread(s['path']), cv2.COLOR_BGR2RGB)
                    axes[row_idx][col_idx].imshow(img)
                    axes[row_idx][col_idx].set_title(label, fontsize=10)
                    axes[row_idx][col_idx].axis('off')

                for col_idx in range(len(samples), n_cols):
                    axes[row_idx][col_idx].axis('off')

            plt.suptitle(f'Samples — {ds}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.graphs_dir}/samples_{self._safe_filename(ds)}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _remove_corrupted(self):
        corrupted = self.df[self.df['height'].isna()]
        print(f"\nCorrupted images: {len(corrupted)}")
        if len(corrupted) > 0:
            print(corrupted[['dataset', 'label', 'image']].to_string(index=False))
            self.df = self.df[self.df['height'].notna()].reset_index(drop=True)
            print(f"Removed. Remaining: {len(self.df)}")
        else:
            print("None found.")

    def _cap_chestxray8_normal(self):
        # Subamostra a classe NORMAL do ChestX-ray8 para não dominar o conjunto.
        dataset, cls = 'ChestX-ray8', 'NORMAL'
        mask         = (self.df['dataset'] == dataset) & (self.df['label'] == cls)
        n_original   = mask.sum()

        if n_original <= self.chestxray8_normal_cap:
            print(f"\n[{dataset}] {cls}: {n_original} samples — below cap, kept as-is.")
            return

        sampled_idx = (
            self.df[mask]
            .sample(self.chestxray8_normal_cap, random_state=self.random_state)
            .index
        )
        self.df = (
            pd.concat([self.df[~mask], self.df.loc[sampled_idx]])
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )

        print(f"\n[{dataset}] {cls}: {n_original} → {self.chestxray8_normal_cap} samples")
        print(f"Final total: {len(self.df)}")
        print(self.df.groupby(['dataset', 'label']).size()
                     .reset_index(name='count').to_string(index=False))

    def preprocess_image(self, path, normalize=True, equalization=None):
        # Uso só em EDA/visualização. No treino os datagens entregam [0,255] crus.
        # normalize=True -> float32 [0,1]; equalization: None | 'hist' | 'adaptive'.
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if equalization is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if equalization == 'hist':
                gray = cv2.equalizeHist(gray)
            elif equalization == 'adaptive':
                gray = self._CLAHE.apply(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_AREA)

        if normalize:
            return img.astype(np.float32) / 255.0
        return img

    def _apply_hist_equalization(self, img):
        # preprocessing_function do datagen: recebe e retorna float32 [0,255].
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        eq     = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB).astype(np.float32)

    def _apply_adaptive_equalization(self, img):
        # Mesmo contrato do hist: [0,255] -> [0,255]; usa o CLAHE compartilhado.
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        eq     = self._CLAHE.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB).astype(np.float32)

    def _plot_preprocessing_example(self):
        # 'after' normalizado p/ [0,1] só para exibição; treino usa [0,255].
        img_orig = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)
        img_proc = self.preprocess_image(self._sample_path)  # [0,1] p/ display

        print(f"\nPreprocessing example — shape: {img_proc.shape} | "
              f"range: [{img_proc.min():.3f}, {img_proc.max():.3f}] "
              f"(display only — training uses [0, 255] + internal preprocess layer)")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_orig)
        axes[0].set_title(f'Original\n{img_orig.shape[1]}×{img_orig.shape[0]}px')
        axes[0].axis('off')
        axes[1].imshow(img_proc)
        axes[1].set_title(
            f'Resized (display, [0,1])\n{self.target_size[1]}×{self.target_size[0]}px\n'
            f'Training input: [0,255] → preprocess layer inside model'
        )
        axes[1].axis('off')

        plt.suptitle('Before × After Resize', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/preprocessing_example.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_equalization_example(self):
        img_orig = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)
        img_hist = self.preprocess_image(self._sample_path, equalization='hist')
        img_adap = self.preprocess_image(self._sample_path, equalization='adaptive')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(axes,
                                  [img_orig, img_hist, img_adap],
                                  ['Original', 'Histogram Eq.', 'CLAHE']):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle('Equalization Comparison', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/equalization_example.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def get_datagen_for_experiment(self, mode: str) -> ImageDataGenerator:
        # Modos: 'baseline' | 'augmented' | 'hist' | 'adaptive'.
        # SEM rescale: datagens entregam [0,255]; o modelo normaliza.
        # preprocessing_function roda antes do rescale e deve retornar float32 [0,255].
        if mode == 'baseline':
            return ImageDataGenerator()

        if mode == 'augmented':
            return ImageDataGenerator(
                rotation_range     = 15,
                width_shift_range  = 0.05,
                height_shift_range = 0.05,
                zoom_range         = 0.1,
                horizontal_flip    = True,
                brightness_range   = [0.85, 1.15],
                fill_mode          = 'nearest',
            )

        if mode == 'hist':
            return ImageDataGenerator(
                preprocessing_function=self._apply_hist_equalization,
            )

        if mode == 'adaptive':
            return ImageDataGenerator(
                preprocessing_function=self._apply_adaptive_equalization,
            )

        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Choose from: 'baseline', 'augmented', 'hist', 'adaptive'."
        )

    def _plot_augmentation_examples(self):
        # Datagen augmented entrega [0,255]; divide por 255 antes do imshow.
        img_uint8       = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)
        sample_expanded = np.expand_dims(img_uint8, axis=0).astype(np.float32)

        aug_gen  = self.get_datagen_for_experiment('augmented')
        aug_iter = aug_gen.flow(sample_expanded, batch_size=1)

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes[0][0].imshow(img_uint8)
        axes[0][0].set_title('Original', fontweight='bold')
        axes[0][0].axis('off')

        for idx in range(1, 10):
            aug_img  = next(aug_iter)[0] / 255.0  # saída [0,255] -> [0,1]
            row, col = divmod(idx, 5)
            axes[row][col].imshow(np.clip(aug_img, 0, 1))
            axes[row][col].set_title(f'Aug {idx}')
            axes[row][col].axis('off')

        plt.suptitle('Data Augmentation Examples', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/augmentation_examples.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _split_kfold(self):
        # 1) holdout estratificado de teste; 2) StratifiedKFold no restante (dev).
        print(f"\nStratifiedKFold split "
              f"(n_splits={self.n_splits}, test_size={self.test_size})...")

        X = self.df['path'].values
        y = self.df['label'].values

        X_dev, self.X_test, y_dev, self.y_test = train_test_split(
            X, y,
            test_size    = self.test_size,
            stratify     = y,
            random_state = self.random_state,
        )

        print(f"\n  Test : {len(self.X_test)} samples "
              f"({len(self.X_test) / len(X) * 100:.1f}%) → {Counter(self.y_test)}")

        skf        = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                     random_state=self.random_state)
        self.folds = []

        print(f"\n  Folds on {len(X_dev)} dev samples:\n")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
            self.folds.append({
                'fold'   : fold_idx,
                'X_train': X_dev[train_idx],
                'y_train': y_dev[train_idx],
                'X_val'  : X_dev[val_idx],
                'y_val'  : y_dev[val_idx],
            })
            print(f"  Fold {fold_idx}: "
                  f"train={len(train_idx)} {dict(Counter(y_dev[train_idx]))} | "
                  f"val={len(val_idx)} {dict(Counter(y_dev[val_idx]))}")

    def _save_splits_csv(self):
        # Salva folds + teste em CSV; fold=-1 marca o teste.
        parts = [pd.DataFrame({'path': self.X_test, 'label': self.y_test,
                                'split': 'test', 'fold': -1})]

        for f in self.folds:
            parts.append(pd.DataFrame({'path': f['X_train'], 'label': f['y_train'],
                                        'split': 'train', 'fold': f['fold']}))
            parts.append(pd.DataFrame({'path': f['X_val'], 'label': f['y_val'],
                                        'split': 'val', 'fold': f['fold']}))

        df_splits = pd.concat(parts, ignore_index=True)
        save_path = 'splits_dataset.csv'
        df_splits.to_csv(save_path, index=False)

        print(f"\nSplits saved to: {os.path.abspath(save_path)}")
        summary = (df_splits[df_splits['fold'] != -1]
                   .groupby(['fold', 'split']).size().reset_index(name='n'))
        print(summary.to_string(index=False))
        print(f"Test set: {len(self.X_test)} samples")
