import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

casas = [{"area": 50, "preco": 240000},{"area": 70, "preco": 320000},{"area": 90, "preco": 400000},
        {"area": 110, "preco": 485000},{"area": 130, "preco": 550000},{"area": 150, "preco": 625000},
        {"area": 170, "preco": 700000},{"area": 190, "preco": 775000},{"area": 210, "preco": 850000},
        {"area": 230, "preco": 925000},{"area": 250, "preco": 1000000},{"area": 270, "preco": 1075000},
        {"area": 290, "preco": 1150000},{"area": 310, "preco": 1225000},{"area": 330, "preco": 1300000},
        {"area": 350, "preco": 1375000},{"area": 370, "preco": 1450000},{"area": 390, "preco": 1525000},
        {"area": 410, "preco": 1600000},{"area": 450, "preco": 1800000},{"area": 480, "preco": 2000000}
        ]

df_casas = pd.DataFrame(casas)

'''
Formula da Regressão Linear:
Y = A * X + B
A = (M * Σ(X*Y) - ΣX * ΣY)/(M * Σ (X^2) - (ΣX)^2)
B = μY - A * μX
'''

# Coletando os valores necessários para a fórmula:
m = len(df_casas)
sum_x = sum(df_casas['area'])
sum_y = sum(df_casas['preco'])
sum_xy = sum(df_casas['area'] * df_casas['preco'])
sum_squared_x = sum(df_casas['area'] ** 2)  
squared_x = sum_x ** 2
mean_x = np.mean(df_casas['area'])
mean_y = np.mean(df_casas['preco'])

# Aplicando os valores no A e B da fórmula
A = (m * sum_xy - sum_x * sum_y) / (m * sum_squared_x - squared_x)
B = mean_y - A * mean_x

'''
Fórmulas para cálculo do erro na Regressão Linear:
MAE = (1/M) * Σ |Yi - Ŷi|
MSE = (1/M) * Σ (Yi - Ŷi)^2
RMSE = sqrt(MSE)
RMSE = sqrt( (1/M) * Σ (Yi - Ŷi)^2 )
R² = 1 - (Σ (Yi - Ŷi)^2 / Σ (Yi - μY)^2)
'''

# Prevendo valores para todo dataframe
df_casas['preco_previsto'] = A * df_casas['area'] + B

# Calculando os erros
MAE = np.mean(np.abs(df_casas['preco'] - df_casas['preco_previsto']))
MSE = np.mean((df_casas['preco'] - df_casas['preco_previsto']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df_casas['preco'] - df_casas['preco_previsto']) ** 2)
SST = sum((df_casas['preco'] - mean_y) ** 2)
R2 = 1 - (SSE / SST)

print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")

# Solicita a área do usuário
X = float(input("Informe a área da casa que deseja saber o preço aproximado: "))
Y = A * X + B  

print(f"O preço aproximado da casa com área {X} m² é: R${Y}")

# Gráfico da regressão linear com erro

# Adicionando pontos reais
plt.figure(figsize=(10,5))
plt.scatter(df_casas['area'], df_casas['preco'], label="Dados Reais", color="blue")  
plt.plot(df_casas['area'], df_casas['preco_previsto'], label="Regressão Linear", color="red")  

# Adicionando linha de erro
for i in range(len(df_casas)):
    plt.vlines(x=df_casas['area'][i], ymin=df_casas['preco_previsto'][i], ymax=df_casas['preco'][i], color='gray', linestyle='dotted')

# Adicionando ponto de previsão do usuário
plt.scatter(X, Y, color='green', s=200, label=f"Resultado para {X}m²", edgecolors='black', zorder=5)  

plt.title("Regressão Linear do Valor das Casas")
plt.xlabel("Área")
plt.ylabel("Preço")
plt.legend()
plt.show()

