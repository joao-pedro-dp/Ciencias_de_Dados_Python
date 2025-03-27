'''
Explicação Matemática

Formula Geral: Y = β0 + β1 * X1 + β2 * X2 + ... + βn * Xn

onde:

Y:  é a variável dependente.
X1, X2, ..., Xn:são as variáveis independentes.
β1, β2, ..., βn: são os coeficientes que estamos tentando encontrar. Esses coeficientes dizem quanto cada variável X contribui para o valor de Y.

Formula dos coeficientes: β = (X^T * X)^-1 * X^T * Y

X: A matriz de variáveis independentes. Cada linha de X representa uma observação (uma linha do DataFrame) e cada coluna representa uma variável (por exemplo, área, quartos, tipo de imóvel, bairro, etc.).
X^T: A transposta de X. Ao fazer XT, estamos trocando as linhas por colunas. Isso é necessário porque o cálculo exige que multipliquemos XT por X.
(X^T * X)^-1: A inversa da matriz X^T * X. Multiplicar uma matriz pela sua inversa é como "desfazer" o efeito da multiplicação inicial.
os erros já foram explicados nas regressões simples e eles não mudam

Explicação Simplificada

Imagine que você quer prever o preço de um imóvel (variável dependente). Para isso, você tem algumas informações sobre o imóvel, como:
Área do imóvel, Número de quartos, Tipo de imóvel, Bairro.
Essas informações são as variáveis independentes que você vai usar para tentar prever o preço. Cada uma dessas informações tem um "peso" que afeta o preço, e o objetico dos cálculos é descobrir esses pesos (coeficientes). 
A equação que usamos para fazer a previsão do preço é uma fórmula onde cada variável (como a área, número de quartos, etc.) tem um "peso" que influencia o preço final.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gerando dados aleatórios
np.random.seed(42)

dados = {
    'Idade': [25, 34, 29, 42, 53, 31, 45, 38, 27, 39, 28, 56, 30, 41, 33, 46, 50, 60, 36, 52],
    'Experiencia': [5, 12, 3, 18, 20, 7, 15, 10, 9, 14, 8, 25, 17, 13, 22, 19, 6, 11, 23, 24],
    'Salario': [3000, 5000, 4500, 8000, 12000, 3500, 7000, 6500, 4000, 5500, 4200, 10000, 7500, 6800, 9000, 11000, 9500, 9800, 4800, 8700],
    'Horas_Trabalho': [40, 44, 38, 45, 50, 42, 41, 43, 39, 48, 36, 47, 49, 40, 44, 42, 41, 50, 46, 43],
    'Nivel_Educacao': ['Superior', 'Pós', 'Médio', 'Superior', 'Pós', 'Médio', 'Superior', 'Médio', 'Médio', 'Superior', 'Médio', 'Superior', 'Pós', 'Superior', 'Médio', 'Pós', 'Superior', 'Médio', 'Médio', 'Pós'],
    'Produtividade': [72.5, 89.1, 65.4, 95.6, 85.7, 77.8, 82.3, 75.4, 69.3, 84.9, 71.2, 92.3, 79.1, 88.4, 86.5, 91.2, 80.8, 78.3, 83.2, 85.1]
}

df = pd.DataFrame(dados)

# Criando os Dummys da coluna Nivel_Educacao e transformando os valores booleanos em 0 e 1
df = pd.get_dummies(df, columns=['Nivel_Educacao'], drop_first=True)
df = df.astype(int)

# o X será todas as colunas menos a variavel que queremos saber
# o Y será a variavel que queremos saber (Produtividade)
X = df.drop('Produtividade', axis=1).values
Y = df['Produtividade'].values

# Obtendo os dados para as formulas
XT = X.T
XTX = XT @ X
XTY = XT @ Y
XTX_inv = np.linalg.pinv(XTX)

# Aplicando os dados na formula
β = XTX_inv @ XTY
print("Coeficientes da regressão:", β)

# Pedindo os valores que o user deseja prever
idade = int(input("Idade: "))
experiencia = int(input("Experiência (anos): "))
salario = int(input("Salário: "))
horas_trabalho = int(input("Horas de trabalho semanais: "))
nivel_educacao = input("Nível de educação (Médio, Superior, Pós): ")
    
# Criar array de entrada
entrada = np.array([idade, experiencia, salario, horas_trabalho, 0, 0])

if nivel_educacao.lower() == 'superior':
    entrada[4] = 1
elif nivel_educacao.lower() == 'pós':
    entrada[5] = 1
    
produtividade_prevista = entrada @ β
print(f"\nProdutividade prevista: {produtividade_prevista:.2f}")

# Gráfico da regressão linear 

# Dados reais vs previstos
plt.scatter(Y, X @ β, color='blue', label='Previsões')  

# Adicionando ponto de previsão do usuário
plt.scatter(produtividade_prevista, produtividade_prevista, color='red', label='Previsão do Usuário', marker='x', s=100)

plt.xlabel("Produtividade Real")
plt.ylabel("Produtividade Estimada")
plt.title("Produtividade Real vs Estimada")
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='black', linestyle='--', label='Linha Ideal') 
plt.legend()
plt.show()

    
    
