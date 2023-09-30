# %% [markdown]
# # Carregar o Conjunto de Dados
# 

# %%
import pandas as pd

# Carregar o conjunto de dados
df_balanced = pd.read_csv('balanced_hate_data.csv')


# %% [markdown]
# # Visualizar a Distribuição das Etiquetas (Classes)
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Colunas corretas para as etiquetas
correct_labels_columns = ['Sexism', 'Women', 'Homophobia', 'Homossexuals', 'Lesbians', 'Body', 'Fat.people', 'Fat.women', 'Ugly.people', 'Ugly.women', 'Racism']

# Visualizar a distribuição das várias etiquetas
plt.figure(figsize=(15, 8))
sns.barplot(x=correct_labels_columns, y=df_balanced[correct_labels_columns].sum().values)
plt.title("Distribuição das Etiquetas")
plt.xlabel("Etiquetas")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# # Visualizar a Distribuição do Comprimento dos Textos
# 

# %%
# Calcular o comprimento dos textos
df_balanced['text_length'] = df_balanced['text'].apply(len)

# Visualizar a distribuição do comprimento dos textos
plt.figure(figsize=(12, 6))
sns.histplot(df_balanced['text_length'], bins=50, kde=False, color='blue')
plt.title("Distribuição do Comprimento dos Textos")
plt.xlabel("Comprimento dos Textos")
plt.ylabel("Frequência")
plt.show()


# %% [markdown]
# # Treinamento 
# ### Pré-processamento e Divisão dos Dados

# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#parametros 
batch_size = 8
# Tokenização e pad sequence
max_words = 500
max_sequence_length = 50

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df_balanced['text'])
sequences = tokenizer.texts_to_sequences(df_balanced['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Divisão dos dados
X = padded_sequences
y = df_balanced[correct_labels_columns].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# # equilibrando pesos

# %%
import numpy as np

# Calculando as frequências das classes
class_freq = np.sum(y_train, axis=0) / y_train.shape[0]

# Calculando os pesos como a inversa da frequência
class_weights = 1 / (class_freq + 1e-5)

# Criando um dicionário de pesos de classe
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    


# %% [markdown]
# # Primeira Topologia: Simples LSTM

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Modelo 1: Simples LSTM
model1 = Sequential([
    Embedding(1000, 32, input_length=max_sequence_length),
    LSTM(32),
    Dense(len(correct_labels_columns), activation='sigmoid')
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.summary()

# Treinamento
history1 = model1.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2, class_weight=class_weight_dict)


# %% [markdown]
# # Segunda Topologia: LSTM + Dense Layer
# ### Nesta topologia, adicionaremos uma camada densa adicional entre a LSTM e a camada de saída.

# %%
# Modelo 2: LSTM + Dense
model2 = Sequential([
    Embedding(1000, 32, input_length=max_sequence_length),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(len(correct_labels_columns), activation='sigmoid')
])

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()

# Treinamento
history2 = model2.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2, class_weight=class_weight_dict)


# %% [markdown]
# # Terceira Topologia: LSTM + Multiple Dense Layers
# ### Nesta topologia, adicionaremos múltiplas camadas densas para tornar o modelo mais complexo.
# 
# 

# %%
# Modelo 3: LSTM + Multiple Dense Layers
model3 = Sequential([
    Embedding(1000, 32, input_length=max_sequence_length),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(correct_labels_columns), activation='sigmoid')
])

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model3.summary()

# Treinamento
history3 = model3.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_split=0.2, class_weight=class_weight_dict)


# %% [markdown]
# # analizando history

# %%
import matplotlib.pyplot as plt

pHistory = history3
# Plotando a curva de perda
plt.figure(figsize=(10, 6))
plt.plot(pHistory.history['loss'], label='Treinamento')
plt.plot(pHistory.history['val_loss'], label='Validação')
plt.title('Curva de Perda do Modelo 1')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotando a curva de acurácia
plt.figure(figsize=(10, 6))
plt.plot(pHistory.history['accuracy'], label='Treinamento')
plt.plot(pHistory.history['val_accuracy'], label='Validação')
plt.title('Curva de Acurácia do Modelo 1')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()


# %% [markdown]
# # teste com Texto :)

# %%
import pandas as pd

# Textos reais para teste
textos_reais = [
    "se a menina chama rayane quando faz 15 anos automaticamente fica grávida",
    "Antes o homossexualismo era proibido.Depois passou a ser tolerado.Hoje é normal.Eu vou embora antes que vire obrigatório",
    "pode ser feia,pode ser gorda... só não pode ser feia é gorda.",
    "Nada contra negros mas EU PREFIRO aquele branquelo palmito cor de leite pra bejar na boca rsrs",
    "nada contra negros mas minha filha não vai trazer bandido pra casa não"
]

# Inicializando uma lista vazia para armazenar os resultados finais
resultados_finais = []

# Loop para percorrer cada texto real
for texto in textos_reais:
    # Tokenização e padronização
    sequencias_reais = tokenizer.texts_to_sequences([texto])
    sequencias_padded = pad_sequences(sequencias_reais, maxlen=max_sequence_length, padding='post', truncating='post')

    # Previsões
    predicao1 = model1.predict(sequencias_padded)
    predicao2 = model2.predict(sequencias_padded)
    predicao3 = model3.predict(sequencias_padded)

    # Inicializando uma lista vazia para armazenar os resultados deste texto
    resultados = []

    # Preenchendo a lista com as previsões
    for i, col in enumerate(correct_labels_columns):
        novo_registro = {
            'Texto': texto,
            'Etiqueta': col,
            'Modelo 1 (%)': round(predicao1[0][i] * 100, 2),
            'Modelo 2 (%)': round(predicao2[0][i] * 100, 2),
            'Modelo 3 (%)': round(predicao3[0][i] * 100, 2)
        }
        resultados.append(novo_registro)

    # Adicionando os resultados deste texto à lista final
    resultados_finais.extend(resultados)

# Transformando a lista de dicionários em um DataFrame
df_resultados_finais = pd.DataFrame(resultados_finais)

# Visualizando o DataFrame
df_resultados_finais



