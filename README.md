# 📱📩 Classificateur SMS (Ham/Spam)

> Un projet **TensorFlow 2 / Keras** qui entraîne un réseau de neurones récurrent (Bi‑LSTM) pour détecter les spams SMS 🕵️‍♂️🚫

---

## 🗂 Arborescence du dépôt

```
.
├── data
│   ├── train-data.tsv
│   └── valid-data.tsv
├── sms_spam_classifier.ipynb  # Notebook complet ⚙️
└── README.md  # Ce fichier 📖
```

---

## 🚀 Guide rapide

```bash
# 1️⃣  Cloner le repo
$ git clone <url-du-dépôt> && cd sms-spam-classifier

# 2️⃣  Créer l'environnement (facultatif)
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 3️⃣  Lancer le notebook Jupyter
$ jupyter notebook sms_spam_classifier.ipynb
```

---

## 🧑‍💻 Étapes clés du notebook

| Étape                            | Description                                                                                                             |              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------ |
| 🔹 **1. Chargement des données** | Lecture des fichiers TSV avec **pandas** → `train_df` & `test_df` + aperçu rapide (`.head()`) 🧐                        |              |
| 🔹 **2. Pré‑traitement**         | Conversion des labels *ham/spam* en entiers, suppression des valeurs manquantes, création des jeux `tf.data.Dataset` 📊 |              |
| 🔹 **3. Batching & Prefetch**    | `shuffle ➜ batch ➜ prefetch` pour optimiser le pipeline ⚡                                                               |              |
| 🔹 **4. TextVectorization**      | Adaptation du vocabulaire (1 000 tokens, séquence max 1 000) 📝                                                         |              |
| 🔹 **5. Construction du modèle** | `Embedding ➜ Bi‑LSTM(64) ➜ Bi‑LSTM(32) ➜ Dense(64) ➜ Dropout ➜ Dense(1)` 🏗️                                            |              |
| 🔹 **6. Entraînement**           | 10 époques, *BinaryCrossentropy* + *Adam* (1e‑4) → \~98 % d’accuracy 💯                                                 |              |
| 🔹 **7. Évaluation**             | `model.evaluate()` sur le jeu de validation 📈                                                                          |              |
| 🔹 **8. Visualisation**          | Fonction helper `plot_graphs` pour suivre *loss* & *accuracy* 🖼️                                                       |              |
| 🔹 **9. Prédiction**             | Fonction `predict_message()` renvoyant \`\[score, "ham"                                                                 | "spam"]\` ✉️ |
| 🔹 **10. Tests unitaires**       | Boucle `test_predictions()` pour valider le modèle ✅                                                                    |              |

---

## 🧠 Concepts ML & NLP en détail 🤓

| Concept                     | Pourquoi ?                                                                                                              | Dans le code                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **TextVectorization** 📝    | Convertit le texte brut en séquences d’index d’un vocabulaire fixé (tokenisation, lower‑case, trim).                    | Couche `TextVectorization` adaptée sur `train_ds`, puis placée en première couche du `Sequential`. |
| **Embedding** 🧩            | Transforme chaque token (entier) en un vecteur dense appris, capturant la sémantique (mots proches ➜ vecteurs proches). | `layers.Embedding(vocab_size, 64, mask_zero=True)` ; `mask_zero` masque les *pads* pour le LSTM.   |
| **Bi‑LSTM** 🔄              | Parcourt la séquence dans les deux sens pour capter le contexte gauche + droit (utile pour le langage).                 | Deux blocs : `Bidirectional(LSTM(64, return_sequences=True))` puis `Bidirectional(LSTM(32))`.      |
| **Masquage** 🙈             | Ignore les zéros (padding) afin qu’ils n’influencent pas la dynamique temporelle.                                       | Automatique grâce à `mask_zero=True` + support natif LSTM.                                         |
| **Dropout** 🚿              | Régularise et évite l’overfitting en « éteignant » aléatoirement des neurones pendant l’entraînement.                   | `layers.Dropout(0.3)` juste avant la couche de sortie.                                             |
| **Binary Cross‑Entropy** 🎯 | Fonction de perte adaptée au classement binaire (ham vs spam) ; attend un logit si `from_logits=True`.                  | `loss=BinaryCrossentropy(from_logits=True)`.                                                       |
| **Adam** ⚙️                 | Optimiseur adaptatif populaire, taux d’apprentissage `1e‑4` pour un entraînement stable.                                | `optimizers.Adam(1e‑4)`.                                                                           |
| **tf.data pipeline** 🚚     | `shuffle ➜ batch ➜ prefetch` maximise l’utilisation du GPU/CPU et lisse le débit.                                       | `train_ds.shuffle().batch().prefetch(AUTOTUNE)`.                                                   |
| **Prefetch** ⚡              | Charge le batch n+1 pendant que le réseau traite le batch n → overlap I/O‑compute.                                      | `prefetch(tf.data.AUTOTUNE)`.                                                                      |

---

## 📊 Résultats

| Metric         | Valeur       |
| -------------- | ------------ |
| Loss (val)     | **≈ 0.05**   |
| Accuracy (val) | **≈ 98.8 %** |

Le classificateur passe tous les tests FreeCodeCamp 🏆.

---

## 🤝 Contribuer

1. Fork ❗
2. Créez une branche `feature/ma‑feature` 🌱
3. Push & PR 🔄

---

## 📄 Licence

MIT © 2024
