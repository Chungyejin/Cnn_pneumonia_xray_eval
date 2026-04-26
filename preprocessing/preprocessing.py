import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPipeline:
    """
    Encapsula as etapas de EDA e pré-processamento do dataset de pneumonia.

    Uso:
        pipeline = DataPipeline(base_path='datasets')
        pipeline.run()
    """

    CLASSES = {'NORMAL', 'PNEUMONIA'}
    CORES   = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336'}

    # Mapeamento explícito porque cada dataset tem uma subpasta com o mesmo nome
    SUBDIRS = {
        'chest_xray'                    : 'chest_xray',
        'ChestX-ray8'                   : 'ChestX-ray8',
        'COVID-19 Image Data Collection': 'COVID-19 Image Data Collection',
    }

    def __init__(self, base_path='datasets', target_size=(224, 224),
                 graphs_dir='graphs', random_state=42):
        self.base_path    = base_path
        self.target_size  = target_size
        self.graphs_dir   = graphs_dir
        self.random_state = random_state

        # Atributos preenchidos durante o run()
        self.df               = None
        self.X_train          = self.X_val = self.X_test = None
        self.y_train          = self.y_val = self.y_test = None
        self.train_datagen    = None
        self.val_test_datagen = None
        self._sample_path     = None  # imagem usada nas visualizações de exemplo

        os.makedirs(self.graphs_dir, exist_ok=True)

    def run(self):
        """Executa o pipeline completo em ordem."""
        self._carregar_metadados()
        self._visao_geral()
        self._plot_distribuicao_classes()
        self._plot_distribuicao_dimensoes()
        self._plot_amostras_visuais()
        self._remover_corrompidas()
        self._plot_preprocessamento()
        self._configurar_augmentation()
        self._plot_augmentation()
        self._dividir_splits()
        self._salvar_csv()

    # Carregamento

    def _normalizar_classe(self, nome_dir):
        # Unifica variações como 'NORMAL (1)' ou 'PNEUMONIA_BACTERIAL' numa label única
        u = nome_dir.upper()
        if 'PNEUMONIA' in u: return 'PNEUMONIA'
        if 'NORMAL'    in u: return 'NORMAL'
        return None  # pastas desconhecidas são ignoradas no loop principal

    def _carregar_metadados(self):
        """
        Percorre os datasets e coleta metadados de cada imagem sem carregá-las
        todas em memória. O DataFrame resultante é usado em todas as etapas seguintes.
        """
        print("Carregando metadados...")
        dados = []

        for dataset in os.listdir(self.base_path):
            # Entra na subpasta interna (ex: chest_xray/chest_xray/)
            dataset_path = os.path.join(self.base_path, dataset,
                                        self.SUBDIRS.get(dataset, dataset))
            if not os.path.isdir(dataset_path):
                continue

            for classe_dir in os.listdir(dataset_path):
                classe_path = os.path.join(dataset_path, classe_dir)
                if not os.path.isdir(classe_path):
                    continue

                classe = self._normalizar_classe(classe_dir)
                if classe is None:
                    continue

                for img_nome in os.listdir(classe_path):
                    if not img_nome.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    caminho = os.path.join(classe_path, img_nome)

                    # Lê a imagem só para extrair dimensões; será None se corrompida
                    img = cv2.imread(caminho)

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

    # EDA

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

    def _plot_distribuicao_classes(self):
        """
        Gera um gráfico de barras por dataset e um extra com o conjunto unificado,
        permitindo identificar visualmente o desbalanceamento entre classes.
        """
        print("\nPlotando distribuição de classes...")
        datasets = self.df['dataset'].unique()
        n        = len(datasets)

        fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

        for ax, ds in zip(axes[:n], datasets):
            contagem = self.df[self.df['dataset'] == ds]['classe'].value_counts()
            self._barplot(ax, contagem, ds)

        # Último subplot mostra todos os datasets combinados
        self._barplot(axes[n], self.df['classe'].value_counts(), 'CONJUNTO UNIFICADO')

        plt.suptitle('Distribuição de Classes por Dataset + Conjunto Unificado',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/distribuicao_classes.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _barplot(self, ax, contagem, titulo):
        """Auxiliar para desenhar barplot com contagem absoluta e percentual."""
        bars  = ax.bar(contagem.index, contagem.values,
                       color=[self.CORES[c] for c in contagem.index],
                       edgecolor='black', linewidth=0.7)
        total = contagem.sum()
        pct   = (contagem / total * 100).round(1)

        # Exibe o valor absoluto acima de cada barra
        for bar, val in zip(bars, contagem.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=10)

        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Quantidade')
        ax.set_xticks(range(len(contagem)))

        # Label do eixo x inclui o percentual para facilitar leitura
        ax.set_xticklabels([f'{c}\n({pct[c]}%)' for c in contagem.index])

    def _plot_distribuicao_dimensoes(self):
        """
        Histogramas de altura e largura por dataset. Imagens com dimensões muito
        variadas confirmam a necessidade do resize no pré-processamento.
        """
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
                # Linha vertical na média para referência rápida
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
                axes = [axes]  # garante que axes seja sempre 2D

            for row_idx, classe in enumerate(classes):
                amostras = self.df[
                    (self.df['dataset'] == ds) &
                    (self.df['classe']  == classe) &
                    (self.df['altura'].notna())  # exclui corrompidas
                ].sample(min(n_cols, 4), random_state=self.random_state)

                for col_idx, (_, s) in enumerate(amostras.iterrows()):
                    # OpenCV lê em BGR; converte para RGB antes de exibir
                    img = cv2.cvtColor(cv2.imread(s['caminho']), cv2.COLOR_BGR2RGB)
                    axes[row_idx][col_idx].imshow(img)
                    axes[row_idx][col_idx].set_title(classe, fontsize=10)
                    axes[row_idx][col_idx].axis('off')

                # Esconde subplots vazios quando a amostra tem menos de n_cols imagens
                for col_idx in range(len(amostras), n_cols):
                    axes[row_idx][col_idx].axis('off')

            plt.suptitle(f'Amostras — {ds}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.graphs_dir}/amostras_{ds.replace(" ", "_")}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # Limpeza

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

    # Pré-processamento

    def preprocessar_imagem(self, caminho, normalizar=True):
        """
        Lê, converte para RGB, redimensiona e normaliza uma imagem.
        Retorna np.ndarray float32 com valores em [0, 1].
        """
        img = cv2.imread(caminho)
        if img is None:
            raise ValueError(f"Imagem não encontrada: {caminho}")

        # OpenCV lê em BGR; CNNs pré-treinadas esperam RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # INTER_AREA é o método mais adequado para redução de resolução
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_AREA)

        # Normaliza para [0, 1]; redes neurais convergem melhor com valores pequenos
        if normalizar:
            img = img.astype(np.float32) / 255.0

        return img

    def _plot_preprocessamento(self):
        """Exibe uma imagem antes e depois do pré-processamento para inspeção visual."""
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

    # Augmentation

    def _configurar_augmentation(self):
        """
        Define dois geradores:
        - train_datagen: aplica transformações aleatórias para aumentar a
          diversidade do treino e reduzir overfitting.
        - val_test_datagen: apenas normaliza, sem transformações, para garantir
          avaliações determinísticas.
        """
        print("\nConfigurando Data Augmentation...")

        self.train_datagen = ImageDataGenerator(
            rescale            = 1./255,
            rotation_range     = 15,    # rotação suave; raio-x tem orientação padrão
            width_shift_range  = 0.05,
            height_shift_range = 0.05,
            zoom_range         = 0.1,
            horizontal_flip    = True,  # flip vertical não faz sentido em raio-x
            brightness_range   = [0.85, 1.15],
            fill_mode          = 'nearest'
        )

        self.val_test_datagen = ImageDataGenerator(rescale=1./255)

    def _plot_augmentation(self):
        """
        Visualiza exemplos de augmentation.
        Importante: o gerador recebe a imagem em uint8 [0, 255] e aplica o
        rescale internamente. Passar float32 [0, 1] causaria imagens pretas
        porque o brightness_range multiplicaria valores já próximos de zero.
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

    # Splits

    def _dividir_splits(self):
        """
        Divide o dataset em treino (70%), validação (15%) e teste (15%).
        O parâmetro stratify garante que a proporção entre classes seja
        mantida igual nos três subconjuntos.
        """
        print("\nDividindo treino / validação / teste...")
        X = self.df['caminho'].values
        y = self.df['classe'].values

        # Primeiro corte: 70% treino, 30% temporário
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=self.random_state
        )

        # Segundo corte: divide o temporário em 50/50 -> 15% val, 15% teste
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp,
            random_state=self.random_state
        )

        total = len(X)
        print(f"Treino    : {len(self.X_train)} ({len(self.X_train)/total*100:.1f}%)")
        print(f"Validação : {len(self.X_val)} ({len(self.X_val)/total*100:.1f}%)")
        print(f"Teste     : {len(self.X_test)} ({len(self.X_test)/total*100:.1f}%)")

        for nome, y_s in [("Treino", self.y_train),
                           ("Validação", self.y_val),
                           ("Teste", self.y_test)]:
            c   = Counter(y_s)
            tot = len(y_s)
            print(f"  {nome}: { {k: f'{v} ({v/tot*100:.1f}%)' for k, v in c.items()} }")

    def _salvar_csv(self):
        """
        Salva os splits em CSV para que a etapa de treinamento possa recarregar
        os mesmos conjuntos sem reexecutar o pré-processamento.
        """
        df_splits = pd.concat([
            pd.DataFrame({'caminho': self.X_train, 'classe': self.y_train, 'split': 'train'}),
            pd.DataFrame({'caminho': self.X_val,   'classe': self.y_val,   'split': 'val'}),
            pd.DataFrame({'caminho': self.X_test,  'classe': self.y_test,  'split': 'test'}),
        ], ignore_index=True)

        save_path = os.path.join(self.base_path, '..', 'splits_dataset.csv')
        df_splits.to_csv(save_path, index=False)
        print(f"\nSplits salvos em: {os.path.abspath(save_path)}")
        print(df_splits['split'].value_counts().to_string())