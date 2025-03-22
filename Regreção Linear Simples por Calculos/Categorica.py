'''
Formula da Regressão Linear Categórica:

Basicamente,a formula é idêntica dicotômica; no entanto, Deve-se criar colunas correspondentes à quantidade de variáveis dummy a serem utilizadas, transformando os valores booleanos em 1 e 0 para usá-los na fórmula.

O valor final obtido pode parecer um pouco estranho, mas o objetivo principal, que eram os cálculos, foi alcançado. 
Para resultados mais precisos, basta manter os cálculos e utilizar uma base de dados real, renomeando as colunas conforme necessário para corresponder às usadas no código e realizar os testes. 
No entanto, como mencionado, meu objetivo foi atingido, então vou encerrar por aqui.

Y = β0 + β1 * D1 + β2 * D2

Onde:
Y: é o preço da casa (variável dependente).
D1: 1 se a observação é de nível "médio" e D1 = 0 caso contrário.
D2: 1 se a observação é de nível "alto" e D2 = 0 caso contrário.
A categoria "baixo" será o grupo de referência (ou seja, quando D1 e D2 = 0.
𝛽0: é o intercepto (o preço médio quando D = 0, ou seja, "Ruim").
𝛽1: é o coeficiente da variável dummy (quanto o preço muda quando D = 1, ou seja, "Bom").
𝛽2: é o coeficiente da variável dummy (quanto o preço muda quando D = 1, ou seja, "Bom").

β1 = (∑(Di1 - μD1) * (Yi - μY)) / (∑(Di1 - D1)^2)

β2 = (∑(Di2 - μD2) * (Yi - μY)) / (∑(Di2 - D2)^2)

Onde:
Di: são os valores da variável dummy.
Yi: são os valores da variável dependente (preço).
μD: é a média de D.
μY: é a média de Y.

β0 = μY - β1 * μD1 - β2 * μD2

Onde:
μD: é a média de D.
μY: é a média de Y.

MAE = (1/M) * Σ |Yi - Ŷi|
MSE = (1/M) * Σ (Yi - Ŷi)^2
RMSE = sqrt(MSE)
RMSE = sqrt( (1/M) * Σ (Yi - Ŷi)^2 )
R² = 1 - (Σ (Yi - Ŷi)^2 / Σ (Yi - μY)^2)

Onde:

M: é o número de observações,
Yi: são os valores reais,
Ŷi: são os valores previstos pelo modelo.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

casas = [{"classe": "Baixo", "preco": 240000},{"classe": "Baixo", "preco": 320000},{"classe": "Baixo", "preco": 400000},
        {"classe": "Baixo", "preco": 485000},{"classe": "Médio", "preco": 550000},{"classe": "Médio", "preco": 625000},
        {"classe": "Médio", "preco": 700000},{"classe": "Médio", "preco": 775000},{"classe": "Médio", "preco": 850000},
        {"classe": "Médio", "preco": 925000},{"classe": "Médio", "preco": 1000000},{"classe": "Alto", "preco": 1075000},
        {"classe": "Alto", "preco": 1150000},{"classe": "Alto", "preco": 1225000},{"classe": "Alto", "preco": 1300000},
        {"classe": "Alto", "preco": 1375000},{"classe": "Alto", "preco": 1450000},{"classe": "Alto", "preco": 1525000},
        {"classe": "Alto", "preco": 1600000},{"classe": "Alto", "preco": 1800000},{"classe": "Alto", "preco": 2000000}]

df = pd.DataFrame(casas)

# Criando uma coluna booleana para o dummy D1
df.loc[df['classe'] != "Médio", 'D1'] = 0
df.loc[df['classe'] == "Médio", 'D1'] = 1

# Criando uma coluna booleana para o dummy D2
df.loc[df['classe'] != "Alto", 'D2'] = 0
df.loc[df['classe'] == "Alto", 'D2'] = 1

# Coletando os valores necessários para D1:
mean_D1 = np.mean(df['D1'])
Di_meanD1= df['D1'] - mean_D1
mean_Y1 = np.mean(df['preco'])
Yi1 = df['preco']
mean_Y_Yi1 = Yi1 - mean_Y1
sum_Di1_meanD1_mean_Y_Yi = np.sum(Di_meanD1 * mean_Y_Yi1)
squared_sum_Di1_meanD1 = np.sum(Di_meanD1**2)

β1 = (sum_Di1_meanD1_mean_Y_Yi / squared_sum_Di1_meanD1)

# Coletando os valores necessários para D2:
mean_D2 = np.mean(df['D2'])
Di_meanD2= df['D2'] - mean_D2
mean_Y2 = np.mean(df['preco'])
Yi2 = df['preco']
mean_Y_Yi2 = Yi2 - mean_Y2
sum_Di2_meanD2_mean_Y_Yi = np.sum(Di_meanD2 * mean_Y_Yi2)
squared_sum_Di2_meanD2 = np.sum(Di_meanD2**2)

β2 = (sum_Di2_meanD2_mean_Y_Yi / squared_sum_Di2_meanD2)

β0 = (mean_Y1 - β1 * mean_D1 - β2 * mean_D2)

# Calculando os erros
df['preco_previsto'] = β0 + β1 * df['D1'] + β2 * df['D2']
MAE = np.mean(np.abs(df['preco'] - df['preco_previsto']))
MSE = np.mean((df['preco'] - df['preco_previsto']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df['preco'] - df['preco_previsto']) ** 2)
SST = sum((df['preco'] - mean_Y1) ** 2)
R2 = 1 - (SSE / SST)

# Input para receber uma avaliação (D) com if básico, pois o objetivo é a regressão linear 
D = int(input("Informe a classe da casa que deseja saber o preço aproximado: (0 - Baixo, 1 - Médio, 2 - Alto): "))

D1 = 1 if D == 1 else 0
D2 = 1 if D == 2 else 0

# Calculando o preço aproximado para a avaliação fornecida
Y = β0 + β1 * D1 + β2 * D2
print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")
print(f"O preço aproximado da casa com avaliação {D} é: R${Y}")

# Gráfico da regressão linear com erro

# Adicionando pontos reais
plt.figure(figsize=(10,5))
plt.scatter(df['D1'] + 2 * df['D2'], df['preco'], label="Dados Reais", color="blue")  
plt.plot(df['D1'] + 2 * df['D2'], df['preco_previsto'], label="Regressão Linear Dicotômica", color="red")  

# Adicionando linha de erro
for i in range(len(df)):
    plt.vlines(x=df['D1'][i] + 2 * df['D2'][i], ymin=df['preco_previsto'][i], ymax=df['preco'][i], color='gray', linestyle='dotted')

# Adicionando ponto de previsão do usuário
plt.scatter(D, Y, color='green', s=200, label=f"Resultado Avaliação {D}", edgecolors='black', zorder=5)  

plt.title("Regressão Linear Categórica do Valor das Casas")
plt.xlabel("Classe")
plt.ylabel("Preço")
plt.legend()
plt.show()

