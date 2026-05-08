import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPipeline:
    """
    Encapsula as etapas de EDA e pré-processamento do dataset de pneumonia.

    Estratégia de divisão:
        1. Um test set fixo (holdout) é separado antes de qualquer CV.
           Ele nunca participa do treinamento nem da validação — serve
           exclusivamente para a avaliação final e imparcial do modelo.
        2. O restante dos dados é dividido via StratifiedKFold em K folds.
           Em cada fold, (K-1) partes formam o treino e 1 parte forma a
           validação, garantindo que toda amostra seja usada para validar
           exatamente uma vez.

    Balanceamento:
        Apenas a classe NORMAL do ChestX-ray8 é amostrada (≈60k → teto
        configurável). Os demais datasets são mantidos intactos.
        O desbalanceamento residual entre classes é tratado via class_weight
        no treinamento, sem descartar amostras válidas dos outros datasets.

    Uso:
        pipeline = DataPipeline(base_path='datasets')
        pipeline.run()
    """

    CLASSES = {'NORMAL', 'PNEUMONIA'}
    CORES   = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336'}

    def __init__(self, base_path='datasets', target_size=(224, 224),
                 graphs_dir='graphs', random_state=42,
                 chestxray8_normal_cap=5000,
                 n_splits=5, test_size=0.15):
        """
        Parâmetros
        ----------
        base_path            : raiz dos datasets
        target_size          : dimensão alvo após resize (H, W)
        graphs_dir           : pasta de saída dos gráficos
        random_state         : semente global para reprodutibilidade
        chestxray8_normal_cap: teto de amostras NORMAL do ChestX-ray8
        n_splits             : número de folds do StratifiedKFold
        test_size            : fração do dataset reservada para o holdout test
        """
        self.base_path             = base_path
        self.target_size           = target_size
        self.graphs_dir            = graphs_dir
        self.random_state          = random_state
        self.chestxray8_normal_cap = chestxray8_normal_cap
        self.n_splits              = n_splits
        self.test_size             = test_size

        self.df               = None
        self.folds            = []
        self.X_test           = None
        self.y_test           = None
        self.train_datagen    = None
        self.val_test_datagen = None
        self._sample_path     = None

        os.makedirs(self.graphs_dir, exist_ok=True)

    def run(self):
        """Executa o pipeline completo em ordem."""
        self._carregar_metadados()
        self._visao_geral()
        self._plot_distribuicao_classes(sufixo='_original')
        self._plot_distribuicao_dimensoes()
        self._plot_amostras_visuais()
        self._remover_corrompidas()
        self._amostrar_estratificado()
        self._plot_distribuicao_classes(sufixo='_balanceado')
        self._plot_preprocessamento()
        self._configurar_augmentation()
        self._plot_augmentation()
        self._dividir_kfold()
        self._salvar_csv()

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def _normalizar_classe(self, nome_dir):
        """
        Unifica variações como 'NORMAL (1)' ou 'PNEUMONIA_BACTERIAL'
        numa label única. Retorna None para pastas intermediárias.
        """
        u = nome_dir.upper()
        if 'PNEUMONIA' in u:
            return 'PNEUMONIA'
        if 'NORMAL' in u:
            return 'NORMAL'
        return None

    def _carregar_metadados(self):
        """
        Percorre os datasets recursivamente usando os.walk, localizando
        pastas NORMAL/PNEUMONIA em qualquer nível — compatível com
        estruturas flat (chest_xray) e aninhadas com múltiplos
        subdiretórios intermediários (ChestX-ray8 com timestamps).
        """
        print("Carregando metadados...")
        dados = []

        for dataset in os.listdir(self.base_path):
            dataset_path = os.path.join(self.base_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for dirpath, _, filenames in os.walk(dataset_path):
                classe = self._normalizar_classe(os.path.basename(dirpath))
                if classe is None:
                    continue

                for img_nome in filenames:
                    if not img_nome.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    caminho = os.path.join(dirpath, img_nome)
                    img     = cv2.imread(caminho)

                    if img is None:
                        altura, largura, canais = None, None, None
                    else:
                        altura, largura = img.shape[:2]
                        canais          = img.shape[2] if len(img.shape) == 3 else 1

                    dados.append({
                        'dataset': dataset, 'classe': classe,
                        'imagem': img_nome, 'caminho': caminho,
                        'altura': altura, 'largura': largura, 'canais': canais
                    })

        self.df = pd.DataFrame(dados)
        print(f"Total carregado: {len(self.df)} imagens")
        print(self.df.groupby(['dataset', 'classe']).size()
                     .reset_index(name='quantidade').to_string(index=False))

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------

    def _visao_geral(self):
        total = self.df.groupby('dataset').size().reset_index(name='total')
        geral = self.df['classe'].value_counts().reset_index()
        geral.columns = ['classe', 'quantidade']

        print("\nTotal por dataset:")
        print(total.to_string(index=False))
        print("\nTotal por classe (conjunto unificado):")
        print(geral.to_string(index=False))
        print(f"\nTotal de imagens    : {len(self.df)}")
        print(f"Imagens corrompidas : {self.df['altura'].isna().sum()}")

    def _plot_distribuicao_classes(self, sufixo=''):
        """
        Gera um gráfico de barras por dataset e um extra com o conjunto
        unificado. O sufixo diferencia versões antes/após a amostragem.
        """
        print(f"\nPlotando distribuição de classes{sufixo}...")
        datasets = self.df['dataset'].unique()
        n        = len(datasets)

        fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

        for ax, ds in zip(axes[:n], datasets):
            contagem = self.df[self.df['dataset'] == ds]['classe'].value_counts()
            self._barplot(ax, contagem, ds)

        self._barplot(axes[n], self.df['classe'].value_counts(), 'CONJUNTO UNIFICADO')

        titulo = 'Distribuição de Classes por Dataset + Conjunto Unificado'
        if sufixo:
            titulo += f' ({sufixo.strip("_").capitalize()})'

        plt.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/distribuicao_classes{sufixo}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _barplot(self, ax, contagem, titulo):
        """Auxiliar para desenhar barplot com contagem absoluta e percentual."""
        bars  = ax.bar(contagem.index, contagem.values,
                       color=[self.CORES[c] for c in contagem.index],
                       edgecolor='black', linewidth=0.7)
        total = contagem.sum()
        pct   = (contagem / total * 100).round(1)

        for bar, val in zip(bars, contagem.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=10)

        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Quantidade')
        ax.set_xticks(range(len(contagem)))
        ax.set_xticklabels([f'{c}\n({pct[c]}%)' for c in contagem.index])

    def _plot_distribuicao_dimensoes(self):
        """Histogramas de altura e largura por dataset."""
        print("\nPlotando distribuição de dimensões...")
        datasets  = self.df['dataset'].unique()
        fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4 * len(datasets)))

        for i, ds in enumerate(datasets):
            df_t = self.df[self.df['dataset'] == ds].dropna(subset=['altura', 'largura'])
            ax_h = axes[i][0] if len(datasets) > 1 else axes[0]
            ax_w = axes[i][1] if len(datasets) > 1 else axes[1]

            for ax, col, color, label in [
                (ax_h, 'altura',  'steelblue',  'Altura'),
                (ax_w, 'largura', 'darkorange', 'Largura'),
            ]:
                ax.hist(df_t[col], bins=30, color=color, edgecolor='black', alpha=0.8)
                ax.set_title(f'[{ds}] {label}')
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Frequência')
                ax.axvline(df_t[col].mean(), color='red', linestyle='--',
                           label=f"Média: {df_t[col].mean():.0f}px")
                ax.legend()

            print(f"\n  {ds}")
            for col in ['altura', 'largura']:
                print(f"   {col.capitalize():7} — "
                      f"min: {df_t[col].min():.0f} | max: {df_t[col].max():.0f} | "
                      f"média: {df_t[col].mean():.1f} | mediana: {df_t[col].median():.0f}")

        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/distribuicao_dimensoes.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_amostras_visuais(self):
        """Exibe uma grade com amostras aleatórias por dataset e classe."""
        print("\nGerando amostras visuais...")
        for ds in self.df['dataset'].unique():
            classes        = self.df[self.df['dataset'] == ds]['classe'].unique()
            n_cols, n_rows = 4, len(classes)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            if n_rows == 1:
                axes = [axes]

            for row_idx, classe in enumerate(classes):
                amostras = self.df[
                    (self.df['dataset'] == ds) &
                    (self.df['classe']  == classe) &
                    (self.df['altura'].notna())
                ].sample(min(n_cols, 4), random_state=self.random_state)

                for col_idx, (_, s) in enumerate(amostras.iterrows()):
                    img = cv2.cvtColor(cv2.imread(s['caminho']), cv2.COLOR_BGR2RGB)
                    axes[row_idx][col_idx].imshow(img)
                    axes[row_idx][col_idx].set_title(classe, fontsize=10)
                    axes[row_idx][col_idx].axis('off')

                for col_idx in range(len(amostras), n_cols):
                    axes[row_idx][col_idx].axis('off')

            plt.suptitle(f'Amostras — {ds}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.graphs_dir}/amostras_{ds.replace(" ", "_")}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Limpeza
    # ------------------------------------------------------------------

    def _remover_corrompidas(self):
        """Remove imagens que o OpenCV não conseguiu abrir (altura == None)."""
        corrompidas = self.df[self.df['altura'].isna()]
        print(f"\nImagens corrompidas: {len(corrompidas)}")
        if len(corrompidas) > 0:
            print(corrompidas[['dataset', 'classe', 'imagem']].to_string(index=False))
            self.df = self.df[self.df['altura'].notna()].reset_index(drop=True)
            print(f"Removidas. Restam {len(self.df)} imagens.")
        else:
            print("Nenhuma imagem corrompida.")

    # ------------------------------------------------------------------
    # Amostragem Estratificada
    # ------------------------------------------------------------------

    def _amostrar_estratificado(self):
        """
        Amostra apenas a classe NORMAL do ChestX-ray8, que concentra
        ~60k imagens e distorceria o conjunto unificado.

        Todos os outros datasets são mantidos intactos, preservando
        suas distribuições originais de classe. O desbalanceamento
        residual é tratado via class_weight no treinamento.
        """
        dataset_alvo = 'ChestX-ray8'
        classe_alvo  = 'NORMAL'

        mascara = (
            (self.df['dataset'] == dataset_alvo) &
            (self.df['classe']  == classe_alvo)
        )

        n_original = mascara.sum()

        if n_original > self.chestxray8_normal_cap:
            idx_amostrados = (
                self.df[mascara]
                .sample(self.chestxray8_normal_cap, random_state=self.random_state)
                .index
            )
            self.df = pd.concat([
                self.df[~mascara],
                self.df.loc[idx_amostrados]
            ]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

            print(f"\nAmostragem: [{dataset_alvo}] {classe_alvo} "
                  f"{n_original} → {self.chestxray8_normal_cap} amostras")
        else:
            print(f"\nAmostragem: [{dataset_alvo}] {classe_alvo} "
                  f"abaixo do teto ({n_original}), mantido.")

        print(f"Total final: {len(self.df)} imagens")
        print(self.df.groupby(['dataset', 'classe']).size()
                     .reset_index(name='quantidade').to_string(index=False))

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------

    def preprocessar_imagem(self, caminho, normalizar=True):
        """
        Lê, converte para RGB, redimensiona e normaliza uma imagem.
        Retorna np.ndarray float32 com valores em [0, 1].
        """
        img = cv2.imread(caminho)
        if img is None:
            raise ValueError(f"Imagem não encontrada: {caminho}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_AREA)

        if normalizar:
            img = img.astype(np.float32) / 255.0

        return img

    def _plot_preprocessamento(self):
        """Exibe uma imagem antes e depois do pré-processamento."""
        print("\nDemonstrando pré-processamento...")
        self._sample_path = self.df['caminho'].iloc[0]
        img_proc = self.preprocessar_imagem(self._sample_path)
        img_orig = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)

        print(f"Shape  : {img_proc.shape}")
        print(f"Min/Max: {img_proc.min():.4f} / {img_proc.max():.4f}")
        print(f"Dtype  : {img_proc.dtype}")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_orig)
        plt.title(f'Original\n{img_orig.shape[1]}×{img_orig.shape[0]}px')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_proc)
        plt.title(f'Pré-processada\n{self.target_size[1]}×{self.target_size[0]}px | [0,1]')
        plt.axis('off')
        plt.suptitle('Antes × Depois do Pré-Processamento', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/preprocessamento_exemplo.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _configurar_augmentation(self):
        """
        Define dois geradores:
        - train_datagen   : transformações aleatórias para diversidade no treino.
        - val_test_datagen: apenas normaliza, para avaliações determinísticas.
        """
        print("\nConfigurando Data Augmentation...")

        self.train_datagen = ImageDataGenerator(
            rescale            = 1./255,
            rotation_range     = 15,
            width_shift_range  = 0.05,
            height_shift_range = 0.05,
            zoom_range         = 0.1,
            horizontal_flip    = True,
            brightness_range   = [0.85, 1.15],
            fill_mode          = 'nearest'
        )

        self.val_test_datagen = ImageDataGenerator(rescale=1./255)

    def _plot_augmentation(self):
        """
        Visualiza exemplos de augmentation.
        O gerador recebe uint8 [0,255] e aplica rescale internamente.
        """
        img_uint8       = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)
        sample_expanded = np.expand_dims(img_uint8, axis=0).astype(np.uint8)

        aug_gen = ImageDataGenerator(
            rescale=1./255, rotation_range=15,
            width_shift_range=0.05, height_shift_range=0.05,
            zoom_range=0.1, horizontal_flip=True,
            brightness_range=[0.85, 1.15], fill_mode='nearest'
        )

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes[0][0].imshow(img_uint8)
        axes[0][0].set_title('Original', fontweight='bold')
        axes[0][0].axis('off')

        aug_iter = aug_gen.flow(sample_expanded, batch_size=1)
        for idx in range(1, 10):
            aug_img  = next(aug_iter)[0]
            row, col = divmod(idx, 5)
            axes[row][col].imshow(np.clip(aug_img, 0, 1))
            axes[row][col].set_title(f'Aug {idx}')
            axes[row][col].axis('off')

        plt.suptitle('Exemplos de Data Augmentation', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/data_augmentation_exemplos.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Divisão com StratifiedKFold
    # ------------------------------------------------------------------

    def _dividir_kfold(self):
        """
        Divisão em duas etapas:

        Etapa 1 — Holdout test set (train_test_split estratificado):
            `test_size` % do dataset total é reservado como test set fixo.
            Ele é separado ANTES do KFold e nunca participa dos folds —
            garante que a avaliação final seja imparcial e independente
            de qualquer decisão tomada durante o treinamento/validação.

        Etapa 2 — StratifiedKFold sobre o restante:
            O restante (1 - test_size) é dividido em `n_splits` folds.
            shuffle=True + random_state fixa a permutação, tornando os
            folds idênticos em toda execução com a mesma semente.

            Em cada fold k:
              - treino   : (n_splits - 1) partes  ≈ (1 - test_size) * (K-1)/K
              - validação: 1 parte                ≈ (1 - test_size) *    1/K

            Exemplo com n_splits=5, test_size=0.15:
              - test      : 15 %
              - treino    : 85 % * 4/5 = 68 %
              - validação : 85 % * 1/5 = 17 %
        """
        print(f"\nDividindo com StratifiedKFold "
              f"(n_splits={self.n_splits}, test_size={self.test_size}, "
              f"random_state={self.random_state})...")

        X = self.df['caminho'].values
        y = self.df['classe'].values

        # --- Etapa 1: holdout test set ---
        X_dev, self.X_test, y_dev, self.y_test = train_test_split(
            X, y,
            test_size    = self.test_size,
            stratify     = y,
            random_state = self.random_state
        )

        total = len(X)
        print(f"\n  Holdout test : {len(self.X_test)} amostras "
              f"({len(self.X_test)/total*100:.1f}%)  "
              f"→ {Counter(self.y_test)}")

        # --- Etapa 2: StratifiedKFold sobre o conjunto de desenvolvimento ---
        skf = StratifiedKFold(
            n_splits     = self.n_splits,
            shuffle      = True,
            random_state = self.random_state
        )

        self.folds = []
        print(f"\n  Folds sobre o conjunto de desenvolvimento "
              f"({len(X_dev)} amostras):\n")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
            fold = {
                'fold'   : fold_idx,
                'X_train': X_dev[train_idx],
                'y_train': y_dev[train_idx],
                'X_val'  : X_dev[val_idx],
                'y_val'  : y_dev[val_idx],
            }
            self.folds.append(fold)

            c_train = Counter(y_dev[train_idx])
            c_val   = Counter(y_dev[val_idx])
            print(f"  Fold {fold_idx}:")
            print(f"    treino    : {len(train_idx):>6} amostras  → {dict(c_train)}")
            print(f"    validação : {len(val_idx):>6} amostras  → {dict(c_val)}")

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _salvar_csv(self):
        """
        Salva os folds e o test set em CSV para reutilização na etapa de
        treinamento sem reexecutar o pipeline.

        Colunas: caminho | classe | split | fold
        fold = -1 para amostras do test set.
        """
        partes = [pd.DataFrame({
            'caminho': self.X_test,
            'classe' : self.y_test,
            'split'  : 'test',
            'fold'   : -1
        })]

        for f in self.folds:
            partes.append(pd.DataFrame({
                'caminho': f['X_train'],
                'classe' : f['y_train'],
                'split'  : 'train',
                'fold'   : f['fold']
            }))
            partes.append(pd.DataFrame({
                'caminho': f['X_val'],
                'classe' : f['y_val'],
                'split'  : 'val',
                'fold'   : f['fold']
            }))

        df_splits = pd.concat(partes, ignore_index=True)

        save_path = os.path.join(self.base_path, '..', 'splits_dataset.csv')
        df_splits.to_csv(save_path, index=False)
        print(f"\nSplits salvos em: {os.path.abspath(save_path)}")

        resumo = (df_splits[df_splits['fold'] != -1]
                  .groupby(['fold', 'split'])
                  .size()
                  .reset_index(name='n'))
        print("\n  Resumo por fold:")
        print(resumo.to_string(index=False))
        print(f"\n  Test set : {len(self.X_test)} amostras (fold = -1)")