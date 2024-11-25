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

# Armazenar histórico de consultas
historico_consultas = []

# Função para pre-processamento dos dados
def preprocess_data(cliente):
    cliente['cpf'] = cliente['cpf'].apply(lambda x: re.sub(r'\D', '', str(x)))
    cliente['trabalha_com_carteira_assinada'] = cliente['trabalha_com_carteira_assinada'].apply(lambda x: 1 if x else 0)
    cliente['inadimplencia_anterior'] = cliente['inadimplencia_anterior'].apply(lambda x: 1 if x else 0).astype('int64')
    cliente['status_pgto_ordem'] = cliente['status_do_pagamento_atual'].map({'Em dia': 1, 'Renegociado': 2, 'Atrasado': 3})
    cliente['score_ajustado'] = cliente.apply(
        lambda row: row['score_de_credito'] * 0.5 if row['inadimplencia_anterior'] == 1 else row['score_de_credito'],
        axis=1
    )
    cliente_features = cliente.drop(columns=['risco_pagamento', 'compra_com_entrada'], errors='ignore')
    cliente_features = cliente_features.select_dtypes(include=['int64', 'float64', 'bool'])
    return cliente_features

# Função para obter nota e feedback do risco
def obter_nota_de_risco(cpf):
    cliente = df[df['cpf'] == cpf]
    if cliente.empty:
        st.warning("CPF não encontrado na base de dados.")
        return None, None
    else:
        cliente_features = preprocess_data(cliente)
        nota_risco = model.predict_proba(cliente_features)[:, 1] * 100
        nota = nota_risco[0]
        
        # Feedback descritivo
        if nota < 40:
            faixa_risco = "Alto Risco"
            descricao = "Cliente com alto risco de inadimplência."
        elif 40 <= nota < 70:
            faixa_risco = "Risco Moderado"
            descricao = "Cliente com risco moderado de inadimplência."
        else:
            faixa_risco = "Baixo Risco"
            descricao = "Cliente com baixo risco de inadimplência."
        
        historico = f"Histórico: {cliente['numero_de_compras_a_credito_anteriores'].values[0]} compras anteriores, "
        historico += "renegociação de compras passada" if cliente['status_do_pagamento_atual'].values[0] == "Renegociado" else "sem renegociação de compras"
        
        feedback = f"{descricao} {historico}"
        historico_consultas.append({"CPF": cpf, "Nota": nota, "Feedback": feedback})
        
        return nota, feedback

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
                {'range': [0, 20], 'color': "rgba(0,255,0,0.8)"},  # Muito Baixo Risco
                {'range': [20, 40], 'color': "rgba(173,255,47,0.8)"},  # Baixo Risco
                {'range': [40, 60], 'color': "rgba(255,255,0,0.8)"},  # Risco Moderado
                {'range': [60, 80], 'color': "rgba(255,165,0,0.8)"},  # Alto Risco
                {'range': [80, 100], 'color': "rgba(255,0,0,0.8)"},   # Muito Alto Risco
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': nota_risco
            }
        }
    ))
    st.plotly_chart(fig)

# Título da aplicação
st.title("Classificação de Risco de Crédito")

# Entrada do CPF
cpf_input = st.text_input("Digite o CPF do cliente (formato: 123.456.789-00):").strip()

# Verificar se o CPF foi inserido no formato correto
if cpf_input:
    if len(cpf_input) == 14 and cpf_input.count(".") == 2 and cpf_input.count("-") == 1:
        nota_de_risco, feedback = obter_nota_de_risco(cpf_input)
        if nota_de_risco is not None:
            st.success(f"A nota de risco do cliente é: {nota_de_risco:.2f}")
            mostrar_gauge(nota_de_risco)
            st.write(feedback)
    else:
        st.error("Por favor, insira o CPF no formato correto (123.456.789-00).")

# Exibir o histórico de consultas
st.subheader("Histórico de Consultas")
for consulta in historico_consultas:
    st.write(f"CPF: {consulta['CPF']}, Nota: {consulta['Nota']:.2f}, Feedback: {consulta['Feedback']}")
