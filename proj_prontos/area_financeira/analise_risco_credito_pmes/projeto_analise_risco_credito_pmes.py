#!/usr/bin/env python
# coding: utf-8

# PROJETO ANALISE DE RISCO DE CRÉDITO EM OPERAÇÕES DE CREDITO/CREDIÁRIO PEQUENAS EMPRESAS

# In[2]:


#!pip install xgboost
# Manipulação de Dados
import pandas as pd
import numpy as np

# Visualização de Dados
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Modelos de Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Avaliação de Modelos
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve
)

# Validação cruzada e ajuste de hiperparâmetros
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Salvando e carregando modelos
import pickle

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

# Definindo o estilo dos gráficos
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# In[3]:


# Carregar o dataset
df = pd.read_csv('dataset_crediario.csv')

# Visualizar as primeiras linhas do dataset para conferir o carregamento
df.head()


# In[4]:


import unidecode

# Renomear as colunas para letras minúsculas, com underscores e sem acentos
df.columns = [
    unidecode.unidecode(col).lower().replace(" ", "_") for col in df.columns
]

# Conferindo as novas colunas
df.head()


# In[5]:


dicionario_colunas = {
    "cpf": "CPF do cliente, identificador único para cada cliente.",
    "nome_completo": "Nome completo do cliente.",
    "profissao": "Profissão do cliente.",
    "trabalha_com_carteira_assinada": "Indica se o cliente possui emprego formal (Sim/Não).",
    "idade": "Idade do cliente em anos.",
    "localizacao": "Cidade de residência do cliente.",
    "renda_mensal": "Renda mensal estimada do cliente em reais.",
    "estado_civil": "Estado civil do cliente (Solteiro, Casado, Divorciado, Viúvo).",
    "numero_de_dependentes": "Número de dependentes do cliente.",
    "score_de_credito": "Pontuação de crédito do cliente, refletindo histórico de crédito.",
    "inadimplencia_anterior": "Indica se o cliente possui histórico de inadimplência (Sim/Não).",
    "valor_total_do_credito_em_aberto": "Total de dívidas em aberto do cliente em reais.",
    "tipo_de_credito_utilizado_anteriormente": "Tipo de crédito que o cliente já utilizou (Pessoal, Veículo, Imobiliário, etc.).",
    "valor_da_compra_a_credito": "Valor total da compra a prazo financiada pelo cliente em reais.",
    "numero_de_parcelas": "Número de parcelas em que a compra foi dividida.",
    "taxa_de_juros_mensal": "Taxa de juros mensal aplicada à operação de crédito (%).",
    "taxa_de_juros_anual": "Taxa de juros anual aplicada à operação de crédito (%).",
    "percentual_de_entrada": "Percentual do valor da compra pago como entrada.",
    "valor_da_entrada": "Valor monetário exato pago como entrada pelo cliente em reais.",
    "compra_com_entrada": "Indica se a compra teve entrada (Sim/Não).",
    "tipo_de_produto_ou_servico_adquirido": "Tipo de produto ou serviço que foi comprado pelo cliente.",
    "data_da_compra": "Data em que a compra foi realizada.",
    "data_de_entrega": "Data em que o produto ou serviço foi entregue ao cliente.",
    "data_da_ultima_compra": "Data da última compra do cliente com a empresa.",
    "data_da_proxima_parcela": "Data de vencimento da próxima parcela da compra.",
    "numero_de_compras_a_credito_anteriores": "Número de compras a crédito que o cliente já fez com a empresa.",
    "montante_total_ja_financiado": "Valor total que o cliente já financiou na empresa em reais.",
    "percentual_de_comprometimento_de_renda": "Percentual da renda mensal do cliente comprometido com outras dívidas.",
    "status_do_pagamento_atual": "Status atual do pagamento (Em dia, Atrasado, Renegociado).",
    "canal_de_aquisicao": "Canal pelo qual o cliente adquiriu o produto/serviço (Presencial, Online, Telefone).",
    "comentarios_e_historico_de_interacoes": "Comentários e histórico de interações com o cliente."
}

# Exibindo o dicionário
dicionario_colunas


# In[6]:


# Quantidade de linhas e colunas
num_linhas, num_colunas = df.shape
print(f"O dataset possui {num_linhas} linhas e {num_colunas} colunas.")

# Visualizar 10 linhas aleatórias do dataset
df.sample(10)


# In[7]:


# Verificar o tipo de dados de cada coluna
df.dtypes


# In[8]:


# Selecionar colunas numéricas
colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
print("Colunas Numéricas:")
print(df[colunas_numericas].head())


# In[9]:


# Selecionar colunas não numéricas (categóricas ou texto)
colunas_nao_numericas = df.select_dtypes(exclude=['int64', 'float64']).columns
print("\nColunas Não Numéricas:")
print(df[colunas_nao_numericas].head())


# ### INICIO AGORA A EXPLORAÇÃO DOS DADOS / LIMPEZA / ORGANIZAÇÃO

# In[11]:


# Transformar a coluna CPF em numérica, removendo caracteres especiais
# df['cpf'] = df['cpf'].str.replace('[^0-9]', '', regex=True).astype(float)

# Transformar a coluna trabalha_com_carteira_assinada em 0 e 1 e converter para numérica
df['trabalha_com_carteira_assinada'] = df['trabalha_com_carteira_assinada'].apply(lambda x: 1 if x else 0)

# Verificar as transformações
df[['cpf', 'trabalha_com_carteira_assinada']].head()


# In[12]:


# Garantir que a coluna CPF seja tratada como string, remover caracteres e converter para inteiro
# df['cpf'] = df['cpf'].astype(str).str.replace('[^0-9]', '', regex=True).astype(int)


# In[13]:


# Exibindo o tipo de dado da coluna 'cpf' para confirmação
cpf_dtype = df['cpf'].dtype
cpf_dtype


# In[14]:


df


# In[15]:


# Exibir uma amostra aleatória da coluna 'inadimplencia_anterior'
df['inadimplencia_anterior'].sample(10)


# In[16]:


# Transformar valores booleanos em binários e mudar o tipo de dado para int64
df['inadimplencia_anterior'] = df['inadimplencia_anterior'].apply(lambda x: 1 if x else 0).astype('int64')


# In[17]:


# Exibir uma amostra aleatória da coluna 'inadimplencia_anterior'
df['inadimplencia_anterior'].sample(10)


# In[18]:


df['status_do_pagamento_atual'].sample(10)


# In[19]:


# Criar uma nova coluna com valores ordinais
df['status_pgto_ordem'] = df['status_do_pagamento_atual'].map({
    'Em dia': 1,
    'Renegociado': 2,
    'Atrasado': 3
})

# Conferir as primeiras linhas para verificar a criação da nova coluna
df[['status_do_pagamento_atual', 'status_pgto_ordem']].head()


# In[20]:


# Selecionar e exibir os nomes das colunas não numéricas ou seja TIPO OBJECT
non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
non_numeric_columns.tolist()


# In[21]:


# Selecionar e exibir os nomes das colunas numéricas
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
numeric_columns.tolist()


# In[22]:


import numpy as np

# Definir condições para a criação da variável alvo
def calcular_risco(row):
    if row['score_de_credito'] < 500 or row['inadimplencia_anterior'] == 1:
        return 0  # Alto risco de inadimplência
    elif row['status_do_pagamento_atual'] == 'Atrasado':
        return 0  # Risco moderado/alto, definido como inadimplente
    elif row['score_de_credito'] >= 700 and row['inadimplencia_anterior'] == 0:
        return 1  # Baixo risco, alta chance de pagamento
    else:
        # Gerar valor aleatório como simulação de risco médio
        return np.random.choice([0, 1], p=[0.3, 0.7])

# Aplicar a função para criar a nova coluna de variável alvo
df['risco_pagamento'] = df.apply(calcular_risco, axis=1)

# Verificar a distribuição da nova variável alvo
df['risco_pagamento'].value_counts(normalize=True)


# In[23]:


# 1. Verificação de Valores Ausentes
valores_ausentes = df.isnull().sum()
valores_ausentes = valores_ausentes[valores_ausentes > 0].sort_values(ascending=False)
percentual_ausentes = (valores_ausentes / len(df)) * 100

print("Valores Ausentes:")
print(pd.DataFrame({'Total de Valores Ausentes': valores_ausentes, 'Percentual (%)': percentual_ausentes}))

# 2. Identificação de Outliers (usando IQR)
outliers = {}
for coluna in df.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    outliers[coluna] = df[(df[coluna] < (Q1 - 1.5 * IQR)) | (df[coluna] > (Q3 + 1.5 * IQR))][coluna]

# Exibir quantidade de outliers por coluna
print("\nOutliers Detectados (por coluna):")
for coluna, valores_outliers in outliers.items():
    print(f"{coluna}: {len(valores_outliers)} outliers")


# In[24]:


df


# In[25]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Filtrar apenas as colunas numéricas e a variável alvo
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['risco_pagamento'])
y = df['risco_pagamento']

# Treinar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Obter a importância das variáveis e convertê-las para percentuais
importancia = rf.feature_importances_ * 100
importancia_percentual = dict(zip(X.columns, importancia))

# Ordenar e exibir a importância das variáveis em percentual
importancia_percentual = {k: v for k, v in sorted(importancia_percentual.items(), key=lambda item: item[1], reverse=True)}
print("Importância das Variáveis (%):")
for variavel, valor in importancia_percentual.items():
    print(f"{variavel}: {valor:.2f}%")

# Plotar o Heatmap de Correlação
plt.figure(figsize=(12, 8))
correlacao = df[X.columns.tolist() + ['risco_pagamento']].corr()
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap de Correlação entre Variáveis Numéricas e Variável Alvo")
plt.show()


# ### Os resultados indicam as variáveis mais influentes para o modelo de classificação de risco:
# 
# #### Principais Variáveis:
# 
# score_de_credito, status_pgto_ordem, e inadimplencia_anterior são as variáveis com maior importância, sugerindo que o histórico de crédito e o comportamento atual de pagamento são fatores cruciais para prever o risco de inadimplência.
# Variáveis Moderadas:
# 
# Variáveis como valor_da_compra_a_credito, montante_total_ja_financiado, e cpf também têm uma influência moderada, possivelmente refletindo o valor total de crédito e o histórico de relacionamento do cliente.
# Variáveis Menos Influentes:
# 
# trabalha_com_carteira_assinada e numero_de_dependentes mostraram pouca importância, indicando que essas características têm um impacto mínimo no risco de crédito para este modelo.

# ## Adicionando novas Features

# In[28]:


# Ajustar a criação da coluna 'score_ajustado' com uma penalização de 50% para clientes com histórico de inadimplência
df['score_ajustado'] = df.apply(
    lambda row: row['score_de_credito'] * 0.5 if row['inadimplencia_anterior'] == 1 else row['score_de_credito'],
    axis=1
)





# In[29]:


# Visualizar uma amostra para verificar os ajustes
df[['score_de_credito', 'inadimplencia_anterior', 'score_ajustado']].sample(10)


# In[30]:


# Visualizar uma amostra de dados da nova coluna 'score_ajustado'
df[['score_ajustado']].sample(5)


# In[31]:


# Filtrar apenas as colunas numéricas e a variável alvo
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['risco_pagamento'])
y = df['risco_pagamento']

# Treinar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Obter a importância das variáveis e convertê-las para percentuais
importancia = rf.feature_importances_ * 100
importancia_percentual = dict(zip(X.columns, importancia))

# Ordenar e exibir a importância das variáveis em percentual
importancia_percentual = {k: v for k, v in sorted(importancia_percentual.items(), key=lambda item: item[1], reverse=True)}
print("Importância das Variáveis (%):")
for variavel, valor in importancia_percentual.items():
    print(f"{variavel}: {valor:.2f}%")


# ## A SEGUIR O TREINAMENTO DO MODELO PARA PREVER O RISCO DA OPERAÇÃO 

# In[33]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error

# Separar os dados em treino e teste
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['risco_pagamento'])
y = df['risco_pagamento']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Prever a variável alvo no conjunto de teste
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probabilidade para métricas contínuas

# Calcular as métricas
mse = mean_squared_error(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred_proba)

# Exibir os resultados
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")


# In[34]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')  # Ignora todos os warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignora FutureWarnings

# Definir os hiperparâmetros para o GridSearch
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Configurar o modelo XGBoost com verbosity=0 para reduzir mensagens de log
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)

# Configurar o GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Executar o GridSearch para encontrar os melhores parâmetros
grid_search.fit(X_train, y_train)

# Exibir os melhores parâmetros e o melhor score encontrado
print("Melhores Hiperparâmetros:", grid_search.best_params_)
print("Melhor F1-Score:", grid_search.best_score_)

# Treinar o modelo com os melhores parâmetros encontrados
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

# Avaliar o modelo otimizado com as métricas solicitadas
mse = mean_squared_error(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred_proba)

# Exibir as métricas do modelo otimizado
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")


# In[35]:


# salvar o novo dataset transformado para deploy

# Definir o caminho para salvar o dataset transformado no mesmo diretório do notebook
novo_dataset_path = './dataset_crediario_transformado.csv'

# Salvar o DataFrame atualizado (transformado) em um novo arquivo CSV
df.to_csv(novo_dataset_path, index=False)

print(f"Novo dataset salvo com sucesso no diretório atual como: {novo_dataset_path}")


# In[36]:


import joblib
import os

# Definir o caminho para salvar o modelo no diretório atual
model_path = os.path.join('./', 'modelo_class_risco_credito.pkl')

# Salvar o modelo otimizado em disco
joblib.dump(best_xgb_model, model_path)

print(f"Modelo salvo em: {model_path}")


# In[ ]:




