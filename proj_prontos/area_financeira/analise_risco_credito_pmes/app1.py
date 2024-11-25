import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import re

# Carregar o modelo treinado
model_path = './modelo_class_risco_credito.pkl'
model = joblib.load(model_path)

# Carregar o dataset transformado na RAM
data_path = './dataset_crediario_transformado.csv'
df = pd.read_csv(data_path)

# Pre-processamento para alinhar os dados com o modelo treinado
def preprocess_data(cliente):
    # Remover pontuações do CPF, se houver
    cliente['cpf'] = cliente['cpf'].apply(lambda x: re.sub(r'\D', '', str(x)))
    
    # Transformações específicas
    cliente['trabalha_com_carteira_assinada'] = cliente['trabalha_com_carteira_assinada'].apply(lambda x: 1 if x else 0)
    cliente['inadimplencia_anterior'] = cliente['inadimplencia_anterior'].apply(lambda x: 1 if x else 0).astype('int64')
    cliente['status_pgto_ordem'] = cliente['status_do_pagamento_atual'].map({
        'Em dia': 1,
        'Renegociado': 2,
        'Atrasado': 3
    })
    cliente['score_ajustado'] = cliente.apply(
        lambda row: row['score_de_credito'] * 0.5 if row['inadimplencia_anterior'] == 1 else row['score_de_credito'],
        axis=1
    )
    
    # Remover colunas não utilizadas pelo modelo
    cliente_features = cliente.drop(columns=['risco_pagamento', 'compra_com_entrada'], errors='ignore')
    
    # Selecionar apenas as colunas numéricas e categóricas necessárias para o modelo
    cliente_features = cliente_features.select_dtypes(include=['int64', 'float64', 'bool'])
    return cliente_features

# Título da aplicação
st.title("Classificação de Risco de Crédito")

# Entrada do CPF no formato correto
cpf_input = st.text_input("Digite o CPF do cliente (formato: 123.456.789-00):").strip()

# Função para obter a nota de risco de crédito
def obter_nota_de_risco(cpf):
    cliente = df[df['cpf'] == cpf]
    if cliente.empty:
        st.warning("CPF não encontrado na base de dados.")
        return None
    else:
        cliente_features = preprocess_data(cliente)
        nota_risco = model.predict_proba(cliente_features)[:, 1] * 100
        return nota_risco[0]

# Função para exibir o gráfico de gauge
def mostrar_gauge(nota_risco):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=nota_risco,
        title={'text': "Nota de Risco de Crédito", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 20], 'color': "rgba(255,0,0,0.8)"},
                {'range': [20, 40], 'color': "rgba(255,165,0,0.8)"},
                {'range': [40, 60], 'color': "rgba(255,255,0,0.8)"},
                {'range': [60, 80], 'color': "rgba(173,255,47,0.8)"},
                {'range': [80, 100], 'color': "rgba(0,255,0,0.8)"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': nota_risco
            }
        }
    ))
    st.plotly_chart(fig)

# Verificar se o CPF foi inserido no formato correto
if cpf_input:
    if len(cpf_input) == 14 and cpf_input.count(".") == 2 and cpf_input.count("-") == 1:
        nota_de_risco = obter_nota_de_risco(cpf_input)
        if nota_de_risco is not None:
            st.success(f"A nota de risco do cliente é: {nota_de_risco:.2f}")
            mostrar_gauge(nota_de_risco)
    else:
        st.error("Por favor, insira o CPF no formato correto (123.456.789-00).")
