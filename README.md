# IA_packages-synthesis
## 1) Types de couches – shapes, rôle, implémentations
### 🔹 MLP (Dense / Fully Connected)

* Rôle

Transformation non linéaire de features

Peut être utilisé instantanément ou par pas de temps

Input shape

Standard : (B, F)

Temporel (sans mélange) : (B, T, F)

Output shape

(B, U)

Temporel : (B, T, U)

⚠️ Point clé (important)

Un MLP appliqué sur (B, T, F) ne mélange pas le temps

Il agit indépendamment sur chaque xₜ

Équivalent à TimeDistributed(MLP)

Keras
```python
Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
)
```
```python
PyTorch
nn.Linear(
    in_features,
    out_features,
    bias=True
)
```
🔹 CNN (Convolutional Neural Network)
CNN 1D (signaux, séries)

Input

Keras : (B, L, C)

PyTorch : (B, C, L)

Output

(B, L', C_out)

Keras
Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    dilation_rate=1,
    activation=None,
    use_bias=True,
)

PyTorch
nn.Conv1d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    bias=True
)

CNN 2D (images)

Input

Keras : (B, H, W, C)

PyTorch : (B, C, H, W)

Keras
Conv2D(
    filters,
    kernel_size,
    strides=(1,1),
    padding="valid",
    activation=None
)

PyTorch
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0
)

🔹 RNN (vanilla)

Rôle

Modélisation séquentielle simple

Dépendances temporelles courtes

Input

(B, T, F)

Output

(B, H) ou (B, T, H)

Keras
SimpleRNN(
    units,
    activation="tanh",
    return_sequences=False,
    return_state=False,
    dropout=0.0,
    recurrent_dropout=0.0
)

PyTorch
nn.RNN(
    input_size,
    hidden_size,
    num_layers=1,
    nonlinearity="tanh",
    batch_first=True,
    dropout=0.0,
    bidirectional=False
)

🔹 LSTM

Rôle

Dépendances longues

Mémoire explicite via cₜ

Input

(B, T, F)

Output

(B, H) ou (B, T, H)

États internes (hₜ, cₜ)

Keras
LSTM(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    return_sequences=False,
    return_state=False,
    dropout=0.0,
    recurrent_dropout=0.0
)

PyTorch
nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    batch_first=True,
    dropout=0.0,
    bidirectional=False
)

🔹 Transformer (Encoder)

Rôle

Dépendances longues sans récurrence

Attention globale

Input

(B, T, D)

Output

(B, T, D)

Keras
MultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0
)


blocs usuels :

LayerNormalization

Dense (FFN)

Skip connections

PyTorch
nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    batch_first=True
)

2) Couches de sortie selon la tâche
🔹 Classification multi-classes (1 classe parmi K)

Sortie : Linear/Dense(K)

Activation : softmax (souvent dans la loss)

Limite : pas multi-label

🔹 Classification binaire

Sortie : Linear(1)

Activation : sigmoid

Limite : sensible au déséquilibre

🔹 Classification multi-label

Sortie : Linear(K)

Activation : sigmoid par classe

Limite : labels supposés indépendants

🔹 Régression non bornée

Sortie : Linear(O)

Activation : aucune

Limite : valeurs physiquement impossibles possibles

🔹 Régression bornée [0,1]

Sortie : Linear(O) + Sigmoid

Limite : saturation proche des bornes

🔹 Régression positive

Sortie : Softplus ou ReLU

Limite : ReLU peut bloquer à 0

3) Losses utilisées dans la littérature
🔹 Régression

MSE : standard, sensible aux outliers

MAE : robuste, convergence plus lente

Huber / SmoothL1 : compromis idéal

NLL Gaussienne : prédiction μ, σ

🔹 Classification

CrossEntropy : multi-classes

Binary Cross Entropy : binaire / multi-label

Focal Loss : classes déséquilibrées

KL Divergence : distributions / distillation

🔹 Séquentiel spécifique

CTC Loss : séquences non alignées

Ranking / Contrastive : embeddings

4) Optimizers – fonctionnement et usages
🔹 SGD

Descente pure du gradient

Bonne généralisation

Lent, LR critique

🔹 SGD + Momentum

Accumulation de vitesse

Très utilisé en CNN vision

🔹 Adam

Moments d’ordre 1 et 2

Rapide, robuste

Standard pour RNN/LSTM

🔹 AdamW

Adam + weight decay correct

Standard pour Transformers

Très bon généraliste

🔹 RMSProp

Moyenne mobile des gradients²

Historiquement utilisé pour RNN

🔹 Adagrad / Adadelta

Features rares

Peu utilisés aujourd’hui

🔹 LAMB / Adafactor

Très gros modèles

NLP / Transformers large-scale

5) Associations typiques observées

CNN (vision) → SGD + momentum / Adam

RNN / LSTM → Adam / RMSProp + gradient clipping

Transformer → AdamW + scheduler + warmup

Régression → Adam(W) + MSE / Huber

Classification → Adam(W) + CE / BCE
