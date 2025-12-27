# IA_packages-synthesis

Conventions :  
B = batch size · T = longueur de séquence · F = features ·  
U = unités · H = hidden size · C = canaux ·  
L = longueur 1D · H/W = hauteur / largeur ·  
D = embedding dim · K = classes · O = output dim

## 1) Types de couches – shapes, rôle, implémentations
### 🔹 MLP (Dense / Fully Connected)

* Rôle

Transformation non linéaire de features

Peut être utilisé instantanément ou par pas de temps

* Input shape

Standard : (B, F)

Temporel (sans mélange) : (B, T, F)

* Output shape

(B, U)

* Temporel : (B, T, U)

⚠️ Point clé (important)

Un MLP pytorch appliqué sur (B, T, F) ne mélange pas le temps

Il agit indépendamment sur chaque xₜ

Avec keras c'est équivalent à TimeDistributed(MLP)
* Implémentation
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
### 🔹 CNN (Convolutional Neural Network)
#### CNN 1D (signaux, séries)

* Input

Keras : (B, L, C)

PyTorch : (B, C, L)

* Output
  
Keras : (B,Lout​,Cout​)

PyTorch : (B,Cout​,Lout​)

* Implémentation

Keras
```python
Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    dilation_rate=1,
    activation=None,
    use_bias=True,
)
```
filters = nombre de filtre et donc = Cout le nombre de canaux en sortie

PyTorch
```python
nn.Conv1d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    bias=True
)
```
#### CNN 2D (images)

* Input

Keras : (B, H, W, C)

PyTorch : (B, C, H, W)

* output

keras : (B,Hout​,Wout​,Cout​)

PyTorch : (B,Cout​,Hout​,Wout​)


Pour faire du time distributed en keras c'est pareil que pour les MLP par contre en pytorch on recommende d'utiliser x.view(B*T,...) pour fuisonner la couche de batch et temporelle( on les remets après mais ainsi le cnn ne mélange pas les infos temporelles).

Dans le cas temporel on a donc :
Tenseurs d’entrée

PyTorch

(B, T, C, H, W)


Keras / TensorFlow

(B, T, H, W, C)
* Implémentation
Keras
```python
Conv2D(
    filters,
    kernel_size,
    strides=(1,1),
    padding="valid",
    activation=None
)
```
PyTorch
```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0
)
```
### 🔹 RNN (vanilla)

* Rôle

Modélisation séquentielle simple

Dépendances temporelles courtes

* Input

(B, T, F)

* Output

(B, H) ou (B, T, H)
* Implémentation
Keras
```python
SimpleRNN(
    units,
    activation="tanh",
    return_sequences=False,
    return_state=False,
    dropout=0.0,
    recurrent_dropout=0.0
)
```
PyTorch
```python
nn.RNN(
    input_size,
    hidden_size,
    num_layers=1,
    nonlinearity="tanh",
    batch_first=True,
    dropout=0.0,
    bidirectional=False
)
```
### 🔹 LSTM

* Rôle

Dépendances longues

Mémoire explicite via cₜ

* Input

(B, T, F)

* Output

(B, H) ou (B, T, H)
en fonction du role qu'on va lui donner :
(B,H,T) si on fait du many-to-many, ex: taging temporel
(B,H) si on fait du many-to-one, ex: la classification
États internes (hₜ, cₜ)
* Implémentation

Keras
-le paramètre return_sequences permet de spécifier si on veux toute la sequence B,T,H ou juste B,H
```python
LSTM(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    return_sequences=False,
    return_state=False,
    dropout=0.0,
    recurrent_dropout=0.0
)
```
PyTorch
-en pytorch il retourne automatiquement toute la sequence l'output a donc d'office la forme B,T,H
```python
nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    batch_first=True,
    dropout=0.0,
    bidirectional=False
)
```
### 🔹 BiLSTM
* input

En général (Keras, et PyTorch avec batch_first=True) :

Input : (B, T, F)

* output

Un BiLSTM concatène forward+backward, donc la dimension cachée devient 2H.

Cas A — sortie à chaque timestep :

Output seq : (B, T, 2H)

Cas B — sortie globale (dernier état) :

Output last : (B, 2H) (souvent on prend le dernier vecteur de la séquence ou on pool)
* Implémentation
Keras

👉 En Keras, un BiLSTM n’est pas une couche séparée, mais un wrapper Bidirectional autour d’un LSTM.

return_sequences garde le même rôle que pour LSTM

la dimension cachée est doublée automatiquement : 2H
```python
Bidirectional(
    LSTM(
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        return_state=False,
        dropout=0.0,
        recurrent_dropout=0.0
    ),
    merge_mode="concat"  # par défaut
)
```

PyTorch

👉 En PyTorch, le BiLSTM est activé via le paramètre bidirectional=True.

PyTorch retourne toujours toute la séquence

la dimension cachée est aussi doublée automatiquement
```python
nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    batch_first=True,
    dropout=0.0,
    bidirectional=True
)
```
### 🔹 ConvLSTM
Un **ConvLSTM** (Convolutional LSTM) est une extension du LSTM classique conçue pour
traiter des **données spatio-temporelles** (séquences d’images, cartes, champs 2D évoluant dans le temps).

L’idée clé est simple :

> **on remplace toutes les opérations fully connected internes du LSTM par des convolutions**.

Ainsi :
- la **structure spatiale** (voisinage, motifs locaux) est conservée,
- la **dynamique temporelle** est modélisée via la mémoire du LSTM.

Conceptuellement, un ConvLSTM combine :
- un **CNN** (pour l’espace),
- un **LSTM** (pour le temps),
mais de manière **couplée et locale**, et non séquentielle.

* input

À chaque pas de temps, l’entrée est une carte spatiale (image / feature map).

En général :

Keras / TensorFlow :

Input : (B, T, H, W, C)


PyTorch :

Input : (B, T, C, H, W)


où :

B : batch size

T : nombre de pas de temps

H, W : dimensions spatiales

C : canaux d’entrée

* output

Un ConvLSTM conserve la structure spatiale dans sa sortie.
La dimension cachée correspond au nombre de filtres convolutionnels F.

Cas A — sortie à chaque timestep :

Output seq : (B, T, H, W, F)


Cas B — sortie finale uniquement :

Output last : (B, H, W, F)


👉 Contrairement à un LSTM/BiLSTM classique, la sortie n’est pas un vecteur, mais une carte 2D (feature map).

* Implémentation

Keras

👉 En Keras, le ConvLSTM est disponible nativement via ConvLSTM2D.

filters joue le rôle de la dimension cachée H

kernel_size définit le voisinage spatial

return_sequences garde le même rôle que pour LSTM
```python
ConvLSTM2D(
    filters,
    kernel_size=(3, 3),
    padding="same",
    activation="tanh",
    return_sequences=False,  # True -> (B, T, H, W, F)
    return_state=False,
    dropout=0.0,
    recurrent_dropout=0.0
)
```
PyTorch

👉 PyTorch ne fournit pas de ConvLSTM natif.
Il faut l’implémenter manuellement ou utiliser une librairie externe.
### 🔹 Transformer (Encoder)

* Rôle

Dépendances longues sans récurrence

Attention globale

Un bloc Transformer Encoder contient exactement :

Multi-Head Self-Attention

Add & Norm

Feed-Forward Network (FFN)

Add & Norm

* Input

(B, T, D)

* Output

(B, T, D)
* Implémentation
  
-Keras
```python
MultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0
)
```
LayerNormalization
```python
self.norm1 = layers.LayerNormalization(epsilon=eps)
```
Dense (FFN)
```python
self.ffn1 = layers.Dense(d_ff, activation=activation)
self.ffn2 = layers.Dense(d_model)
(on essaie toujours de mettre au moins deux couches de FFN)
```
Le forward ressemblera typiquement à :
```python
attn = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
x = self.norm1(x + attn)   # Add & Norm
# Feed-forward
ffn = self.ffn2(self.ffn1(x))
x = self.norm2(x + ffn)
```
⚠️ : keras ne fournis pas instinctivement le positional encoding il faut le rajouter nous même avant de ffaire rentrer l'embedding dans le modèle.


-PyTorch
d_model = taille d'embedding
```python
nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    batch_first=True
)
```
on rajoute une couche de positionnal encoding, du dropout et une linear à la fin :
```python
encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6
)
```

## 2) Couches de sortie selon la tâche
### 🔹 Classification multi-classes (1 classe parmi K)

Sortie : Linear/Dense(K)

Activation : softmax (souvent dans la loss)

Limite : pas multi-label

### 🔹 Classification binaire

Sortie : Linear(1)

Activation : sigmoid

Limite : sensible au déséquilibre

### 🔹 Classification multi-label

Sortie : Linear(K)

Activation : sigmoid par classe

Limite : labels supposés indépendants

### 🔹 Régression non bornée

Sortie : Linear(O)

Activation : aucune

Limite : valeurs physiquement impossibles possibles

### 🔹 Régression bornée [0,1]

Sortie : Linear(O) + Sigmoid

Limite : saturation proche des bornes

### 🔹 Régression positive

Sortie : Softplus ou ReLU

Limite : ReLU peut bloquer à 0

## 3) Losses utilisées dans la littérature
### 🔹 Régression

MSE : standard, sensible aux outliers

MAE : robuste, convergence plus lente

Huber / SmoothL1 : compromis idéal

NLL Gaussienne : prédiction μ, σ

### 🔹 Classification

CrossEntropy : multi-classes

Binary Cross Entropy : binaire / multi-label

Focal Loss : classes déséquilibrées

KL Divergence : distributions / distillation

### 🔹 Séquentiel spécifique

CTC Loss : séquences non alignées

Ranking / Contrastive : embeddings

## 4) Optimizers – fonctionnement et usages
### 🔹 SGD

Descente pure du gradient

Bonne généralisation

Lent, LR critique

### 🔹 SGD + Momentum

Accumulation de vitesse

Très utilisé en CNN vision

### 🔹 Adam

Moments d’ordre 1 et 2

Rapide, robuste

Standard pour RNN/LSTM

### 🔹 AdamW

Adam + weight decay correct

Standard pour Transformers

Très bon généraliste

### 🔹 RMSProp

Moyenne mobile des gradients²

Historiquement utilisé pour RNN

### 🔹 Adagrad / Adadelta

Features rares

Peu utilisés aujourd’hui

### 🔹 LAMB / Adafactor

Très gros modèles

NLP / Transformers large-scale
Dropout — fiche pratique

## 5) Dropout

Technique de régularisation.

Pendant l’entraînement, une fraction p des activations est mise à zéro aléatoirement.

Objectif : réduire l’overfitting en empêchant la co-adaptation des neurones.

Inactif en inference (test).

### 1) Paramètre clé

dropout_rate = p avec p ∈ [0,1)

p = 0.1 → 10 % des activations annulées

p = 0.5 → 50 % annulées

Bonnes valeurs usuelles :

Transformers : 0.1

CNN : 0.2 – 0.5

RNN / LSTM : 0.1 – 0.3

MLP : 0.3 – 0.5

### 2) Implémentation en Keras
#### a) Dropout classique (MLP, CNN, Transformer)
from tensorflow.keras.layers import Dropout

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)


Appliqué sur les activations

Actif uniquement quand training=True

#### b) Dropout dans un Transformer (après sous-blocs)
attn_out = MultiHeadAttention(...)(x, x)
attn_out = Dropout(0.1)(attn_out)
x = LayerNormalization()(x + attn_out)

#### c) RNN / LSTM (spécifique)
LSTM(
    units=128,
    dropout=0.2,           # sur les entrées
    recurrent_dropout=0.0 # sur les connexions récurrentes (souvent 0)
)

### 3) Implémentation en PyTorch
#### a) Dropout classique
import torch.nn as nn

drop = nn.Dropout(p=0.3)

x = self.fc(x)
x = drop(x)

#### b) Dans un nn.Module
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc(x)
        x = self.drop(x)
        return x

#### c) RNN / LSTM
nn.LSTM(
    input_size=F,
    hidden_size=H,
    num_layers=2,
    dropout=0.2,      # entre couches (pas sur les récurrences internes)
    batch_first=True
)


⚠️ En PyTorch, le dropout du LSTM s’applique entre les couches, pas à l’intérieur d’une seule couche.

### 4) Où placer le Dropout (règles simples)
✅ Bonnes pratiques

Après une couche Dense / Linear

Après un sous-bloc Transformer (MHA, FFN), avant Add & Norm

Avant la tête de sortie, jamais après l’activation finale

❌ À éviter

Juste avant une sortie softmax/sigmoid

Trop tôt dans le réseau (perte d’information)

Trop élevé dans les RNN (instabilité temporelle)

### 5) Contextes d’utilisation
#### MLP

Très efficace

Souvent après chaque couche Dense

#### CNN

Utile surtout après les couches denses

Parfois remplacé par SpatialDropout

#### RNN / LSTM

À utiliser avec parcimonie

Plutôt sur les entrées / entre couches

#### Transformer

Standard dans :

MHA

FFN

chemins résiduels

Valeur canonique : 0.1

### 6) Ce que fait / ne fait pas le Dropout

Fait

Réduit l’overfitting

Force des représentations robustes

Améliore la généralisation

Ne fait pas

Ne supprime pas des neurones définitivement

Ne modifie pas l’architecture

N’agit pas en inference

## 5) Associations typiques observées

CNN (vision) → SGD + momentum / Adam

RNN / LSTM → Adam / RMSProp + gradient clipping

Transformer → AdamW + scheduler + warmup

Régression → Adam(W) + MSE / Huber

Classification → Adam(W) + CE / BCE
## 6) éstimer le nombre de couches nécessaires
### Estimation du nombre de couches nécessaires

* Principe fondamental
Il n’existe pas de formule exacte pour déterminer le nombre de couches d’un réseau.
La profondeur doit être choisie **en fonction de la structure du problème**, et non
uniquement à partir des dimensions d’entrée ou de sortie.

La profondeur permet de **factoriser la complexité** :
- couches basses : motifs simples,
- couches intermédiaires : structures composées,
- couches hautes : concepts abstraits.

---

### Heuristique 1 — Complexité spatiale et structure des données

* Données simples
- signaux peu structurés,
- faible variabilité.

👉 1 à 2 couches suffisent.

* Données structurées
- textures,
- motifs répétitifs,
- corrélations locales.

👉 3 à 5 couches sont généralement nécessaires.

* Données très hiérarchiques
- structures complexes,
- dépendances multi-échelles.

👉 5 à 10 couches ou plus, souvent avec connexions résiduelles.

---

### Heuristique 2 — Étendue des dépendances temporelles

* Dépendances courtes
- variations locales,
- peu de mémoire nécessaire.

👉 Convolutions et pooling temporel suffisants.

* Dépendances moyennes
- évolution progressive,
- transitions temporelles claires.

👉 Une couche LSTM ou BiLSTM.

* Dépendances longues
- contexte global important,
- mémoire sur de nombreux pas de temps.

👉 Plusieurs couches récurrentes, ou architectures à attention.

---

### Heuristique 3 — Taille du jeu de données

* Peu de données
- risque élevé de sur-apprentissage.

👉 Réseau peu profond et fortement régularisé.

* Beaucoup de données
- grande diversité,
- meilleure généralisation possible.

👉 Réseau plus profond, avec normalisation et régularisation adaptées.

---

### Heuristique 4 — Nature de la sortie

* Sortie simple
- classification globale,
- régression scalaire.

👉 Peu de couches nécessaires.

* Sortie complexe
- prédiction par pas de temps,
- sorties structurées.

👉 Plus de couches pour capter des relations fines.

---

### Méthode pratique recommandée

1. Commencer par une **architecture simple**.
2. Observer les **courbes d’apprentissage**.
3. Ajouter des couches uniquement en cas de sous-apprentissage.
4. Arrêter l’augmentation de profondeur dès que le gain devient marginal.

---

* Points importants

- Ajouter des couches augmente la capacité, mais aussi le risque d’overfitting.
- La profondeur n’est utile que si elle correspond à une structure réelle dans les données.
- La validation empirique reste indispensable.

---

* Résumé

> Le nombre de couches d’un réseau doit être choisi de manière progressive et justifiée,
> en fonction de la complexité des motifs à apprendre, des dépendances temporelles et
> de la quantité de données disponibles.
