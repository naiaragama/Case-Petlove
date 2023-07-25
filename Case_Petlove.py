#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importando as bibliotecas necessárias para análise
import pandas as pd
import pyspark as ps
import numpy as np


# In[2]:


from matplotlib import pyplot as plt
from matplotlib import figure as f


# In[3]:


df = pd.read_csv('data-test-analytics_5.csv')

df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


#Colocar o id como index do dataset
df.set_index('id',inplace=True)

df.head()


# In[7]:


#elimindando as colunas que não serão relevantes para análise
df.drop(columns= {'name_hash', 'email_hash', 'address_hash','neighborhood'}, inplace=True)

df.head()


# In[8]:


#criando um backup
df_back = df.copy()


# In[9]:


#analisando as colunas em detalhe
df.columns


# In[10]:


#Identificação dos valores nulos
df.isnull().sum()


# In[11]:


# Filtrando os valores nulos ou ausentes | NA , NULL , NaN , NaT 
filtronulo = df.deleted_at.isna()
df.loc[filtronulo]


# In[12]:


#filtro de cancelados
filtrocancelado = df.status == 'canceled'
dfcancelado = df.loc[filtrocancelado]
dfcancelado


# In[13]:


cancelado = df.loc[filtrocancelado]


# In[14]:


#Média de ticket médio, itens comprados, total de receita, total de pedidos 
dfcancelado.mean()


# In[15]:


#Detalhamento dos cancelamentos
dfcancelado.describe()


# In[16]:


#Analisando a situação das assinaturas
df.groupby('status').size().sort_values(ascending=False)


# In[17]:


#Assinaturas por estado
df.groupby(['state'],dropna=False).size().sort_values(ascending=False)


# In[18]:


#Indice de cancelamento por canais
cancelado.groupby('marketing_source').size().sort_values(ascending=False)


# In[20]:


#Plotagem grárfico de barra, notamos maior incidencia de cancelamento por busca orgânica
cancelado.groupby(['marketing_source'],dropna=False).size().sort_values(ascending=False).head(5).plot.bar(figsize=(12,8),xlabel='Canal',ylabel='Qtd. Ocorrencias')


# In[21]:


#indice de cancelamento por estado, possuimos maior indíce em RS
cancelado.groupby('state').size().sort_values(ascending=False)


# In[22]:


#Plotagem grárfico de barra cancelamento por estado, notamos uma média aproximada
cancelado.groupby(['state'],dropna=False).size().sort_values(ascending=False).head(15).plot.bar(figsize=(12,8),xlabel='Estado',ylabel='Qtd. Ocorrencias')


# In[23]:


#filtro de pausados
filtropausado = df.status == 'paused'
dfpausado = df.loc[filtropausado]
dfpausado


# In[24]:


pausado = df.loc[filtropausado]


# In[25]:


#indice de pausas por estado, possuimos maior indíce em MS
pausado.groupby('state').size().sort_values(ascending=False)


# In[26]:


#Plotagem grárfico de barra assinatura pausadas por estado
pausado.groupby(['state'],dropna=False).size().sort_values(ascending=False).head(15).plot.bar(figsize=(12,8),xlabel='Estado',ylabel='Qtd. Ocorrencias')


# In[27]:


#Transformando as datas em datetime
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
df['deleted_at'] = pd.to_datetime(df['deleted_at'], errors='coerce')
df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
df['last_date_purchase'] = pd.to_datetime(df['last_date_purchase'], errors='coerce')


# In[28]:


#Validando as conversões
df.dtypes


# In[29]:


df.head()


# In[30]:


#Compras realizadas no dia do aniversário
df.groupby('birth_date')['recency'].size().sort_values(ascending=False)


# In[31]:


#Tempo médio de cancelamento
df['tempo_cancelamento'] = df['deleted_at'] - df['created_at']
tempo_medio_cancelamento = df.loc[df['status'] == 'canceled', 'tempo_cancelamento'].mean()
print(tempo_medio_cancelamento)


# In[32]:


#Tempo médio entre último pedido e cancelamento da assinatura
df['cancelamento_ultimo_pedido'] = df['deleted_at'] - df['last_date_purchase']
tempo_cancelamento_ultimo_pedido = df.loc[df['status'] == 'canceled', 'cancelamento_ultimo_pedido'].mean()
print(tempo_cancelamento_ultimo_pedido)


# In[33]:


#Cancelamentos diários
df['cancelamento'] = df['deleted_at'].dt.date
cancelamentos_diario = df.groupby(['cancelamento']).size()
cancelamentos_diario


# In[34]:


df.groupby(['cancelamento']).size().head(20).sort_values(ascending=False).plot.bar(xlabel='Qtde Cancelamentos.',ylabel='Data',rot=90)


# In[35]:


df.to_csv('df_churn_final.csv')


# In[ ]:


df['idade'] = datetime.da - df['last_date_purchase']
tempo_cancelamento_ultimo_pedido = df.loc[df['status'] == 'canceled', 'cancelamento_ultimo_pedido'].mean()
print(tempo_cancelamento_ultimo_pedido)

