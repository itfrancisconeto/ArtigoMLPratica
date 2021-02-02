#importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import dump, load
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#titulo
print()
print ('<<< Prever a temperatura máxima dada a temperatura minima >>>')
print()

#carregar dados para o dataframe
df_weather=pd.read_csv(r'Summary_of_Weather.csv', sep=',', usecols=['MaxTemp', 'MinTemp'])

#verificar o shape
print("Dimensões do dataframe:")
print (df_weather.shape)
print()

#verificar os tipos dos dados
print("Tipos dos dados:")
df_weather.info()
print()

print("Primeiras instâncias:")
print(df_weather.head())
print()

#verificar quantidade  de instancias nulas por atributo
print("Atributos com dados faltantes:")
nans = df_weather.isna().sum()
print(nans[nans > 0])
print()

#grafico de correlacao
print ("Gráfico de correlação:")
plt.figure(figsize=(6,5))
df_corr_ = df_weather.corr(method='pearson')
ax_ = sns.heatmap(df_corr_, annot=True)
bottom, top = ax_.get_ylim()
ax_.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
print()

#grafico de dispersao
print ("Gráfico de dispersão:")
plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(df_weather['MinTemp'], df_weather['MaxTemp'], color='gray')
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.show()

#realizar a analise de regressao
X=df_weather['MinTemp'].values.reshape(-1, 1) #variável independente 
y=df_weather['MaxTemp'].values.reshape(-1, 1) #variável dependente

print("Dimensões de X e y:")
print("X:", X.shape, " y:", y.shape)
print()

#dividir o conjunto de dados em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3)

print("Dimensões de X e y para treino e teste:")
print("X_treinamento:", X_treinamento.shape, " X_teste:", X_teste.shape)
print("y_treinamento:", y_treinamento.shape, " y_teste:", y_teste.shape)
print()

#construir modelo de regressao
regressor = LinearRegression()
modelo = regressor.fit (X_treinamento, y_treinamento)

#realizar a previsao
previsao = modelo.predict(X_teste)

#analisar a qualidade do modelo
R_2 = r2_score(y_teste, previsao)  #realiza o cálculo do R2
print ("Coeficiente R2 (%):", R_2.round(4) * 100)
print()

#grafico de dispersao
print ("Gráfico de dispersão:")
plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(X_treinamento, y_treinamento,  color='gray')
plt.plot(X_teste, previsao, color='red')
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.show()

#exportar o modelo para arquivo
dump(modelo, 'previsor.pkl')

# #importar o modelo para memoria e testar
# modelo = load('previsor.pkl')
# X = np.array([[12]]).astype(np.float64)
# previsao=modelo.predict(X)
# previsao = np.round(previsao, 2) 
# print(previsao)
