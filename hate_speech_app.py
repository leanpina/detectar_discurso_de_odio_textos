import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# Função para prever o discurso de ódio
def predict_hate_speech(text, model, tokenizer, max_sequence_length=50):
    # Tokenização e padronização
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Previsão
    prediction = model.predict(padded_sequences)[0]
    
    return prediction

# Função para colorir as células
def color_cells(val):
    color = f'rgb({255 - val * 2.55}, {val * 2.55}, 0)'
    return f'background-color: {color};'

# Carregar o modelo e o tokenizador
model = load_model('best_hate_speech_model.h5')
with open('hate_speech_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Colunas corretas para as etiquetas
correct_labels_columns = ['Sexismo', 'Mulheres', 'Homofobia', 'Homossexuais', 'Lesbicas', 'Corpo', 'Pessoas gordas', 'Mulheres gordas', 'Pessoas feias', 'Mulheres feias', 'Racismo']

# Interface do Streamlit
st.title('Análise de Discurso de Ódio | INFNET')
user_input = st.text_input("Digite um texto para analisar:")

if user_input:
    prediction = predict_hate_speech(user_input, model, tokenizer)
    
    # Usar st.columns para criar duas colunas lado a lado
    col2, col1 = st.columns(2)
    
    # Na primeira coluna, mostrar o resumo do modelo
    with col1:
        st.markdown("Resumo do Modelo:")
        st.text("Modelo Carregado: Sequential")
        st.text("Topologia:")
        st.text("Camada de Entrada (Embedding)")
        st.text("Camada LSTM 1")
        st.text("Camada LSTM 2")
        st.text("Camada Densa")
        st.text("Camada de Saída")
    
    # Na segunda coluna, mostrar a tabela
    with col2:
        # Criando um DataFrame para exibir os resultados
        df_result = pd.DataFrame({
            'Etiqueta': correct_labels_columns,
            'Probabilidade (%)': (prediction * 100).round(2)
        })

        # Ordenando o DataFrame pela probabilidade
        df_result = df_result.sort_values(by='Probabilidade (%)', ascending=False)

        # Renderizando o DataFrame com cores
        st.dataframe(df_result.style.applymap(color_cells, subset=['Probabilidade (%)']))

