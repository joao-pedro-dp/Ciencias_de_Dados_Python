'''
Formula da Regressão Linear Dicotômica:

Basicamente,a formula é quase idêntica à numérica; no entanto, deve-se criar uma coluna transformando os valores booleanos em 1 e 0 para usá-los na fórmula.

Y = β0 + β1 * D 

Onde:
Y: é o preço da casa (variável dependente).
D: é a variável dicotômica (0 ou 1).
𝛽0: é o intercepto (o preço médio quando D = 0, ou seja, "Ruim").
𝛽1: é o coeficiente da variável dummy (quanto o preço muda quando D = 1, ou seja, "Bom").

β1 = (∑(Di - μD) * (Yi - μY)) / (∑(Di - D)^2)

Onde:
Di: são os valores da variável dummy.
Yi: são os valores da variável dependente (preço).
μD: é a média de D.
μY: é a média de Y.

β0 = μY - β1 * μD

Onde:
B1: é a Formula de cima
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

casas = [{"avaliacao": "Ruim", "preco": 240000},{"avaliacao": "Bom", "preco": 320000},{"avaliacao": "Ruim", "preco": 400000},
        {"avaliacao": "Bom", "preco": 485000},{"avaliacao": "Ruim", "preco": 550000},{"avaliacao": "Bom", "preco": 625000},
        {"avaliacao": "Bom", "preco": 700000},{"avaliacao": "Bom", "preco": 775000},{"avaliacao": "Bom", "preco": 850000},
        {"avaliacao": "Bom", "preco": 925000},{"avaliacao": "Bom", "preco": 1000000},{"avaliacao": "Ruim", "preco": 1075000},
        {"avaliacao": "Bom", "preco": 1150000},{"avaliacao": "Bom", "preco": 1225000},{"avaliacao": "Bom", "preco": 1300000},
        {"avaliacao": "Ruim", "preco": 1375000},{"avaliacao": "Bom", "preco": 1450000},{"avaliacao": "Ruim", "preco": 1525000},
        {"avaliacao": "Bom", "preco": 1600000},{"avaliacao": "Bom", "preco": 1800000},{"avaliacao": "Bom", "preco": 2000000}
        ]

df = pd.DataFrame(casas)

# Criando uma coluna booleana dummy com base na avaliação
df.loc[df['avaliacao'] == "Ruim", 'D'] = 0
df.loc[df['avaliacao'] == "Bom", 'D'] = 1

# Coletando os valores necessários para as fórmulas:
mean_D = np.mean(df['D'])
Di_meanD= df['D'] - mean_D
mean_Y = np.mean(df['preco'])
Yi = df['preco']
mean_Y_Yi = Yi - mean_Y
sum_Di_meanD_mean_Y_Yi = np.sum(Di_meanD * mean_Y_Yi)
squared_sum_Di_meanD = np.sum(Di_meanD**2)

β1 = (sum_Di_meanD_mean_Y_Yi / squared_sum_Di_meanD)

β0 = (mean_Y - β1 * mean_D)

# Calculando os erros
df['preco_previsto'] = β0 + β1 * df['D']
MAE = np.mean(np.abs(df['preco'] - df['preco_previsto']))
MSE = np.mean((df['preco'] - df['preco_previsto']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df['preco'] - df['preco_previsto']) ** 2)
SST = sum((df['preco'] - mean_Y) ** 2)
R2 = 1 - (SSE / SST)

# Input para receber uma avaliação (D)
D = int(input("Informe a avaliação da casa que deseja saber o preço aproximado (0 para 'Ruim' e 1 para 'Bom'): "))

# Calculando o preço aproximado para a avaliação fornecida
Y = β0 + β1 * D

print(f"O preço aproximado da casa com avaliação {D} (0 para 'Ruim' ou 1 para 'Bom') é: R${Y}")
print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")
print(f"O preço aproximado da casa com avaliação {D} é: R${Y}")

# Gráfico da regressão linear com erro

# Adicionando pontos reais
plt.figure(figsize=(10,5))
plt.scatter(df['D'], df['preco'], label="Dados Reais", color="blue")  
plt.plot(df['D'], df['preco_previsto'], label="Regressão Linear Dicotômica", color="red")  

# Adicionando linha de erro
for i in range(len(df)):
    plt.vlines(x=df['D'][i], ymin=df['preco_previsto'][i], ymax=df['preco'][i], color='gray', linestyle='dotted')

# Adicionando ponto de previsão do usuário
plt.scatter(D, Y, color='green', s=200, label=f"Resultado Avaliação {D}", edgecolors='black', zorder=5)  

plt.title("Regressão Linear Dicotômica do Valor das Casas")
plt.xlabel("Avaliação")
plt.ylabel("Preço")
plt.legend()
plt.show()
