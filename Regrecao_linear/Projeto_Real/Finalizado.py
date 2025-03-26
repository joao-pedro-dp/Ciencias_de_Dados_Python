import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df = pd.read_csv('usina.csv')
df = df.rename(columns={'AT': 'Temperatura', 'V': 'Pressão do Ar','RH': 'Umidade do Ar','AP': 'Pressão ATM','PE': 'Energia Produzida',})

y = df['Energia Produzida']
X = df.drop(columns='Energia Produzida')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

df_train = pd.DataFrame(data= X_train)
df_train['Energia Produzida'] = y_train

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

modelo = sm.OLS(y_train,
                  X_train[['const', 'Pressão do Ar', 'Pressão ATM', 'Umidade do Ar']]).fit()

predict = modelo.predict(X_test[['const', 'Pressão do Ar', 'Pressão ATM', 'Umidade do Ar']])

novo_dado = pd.DataFrame({ 'const': [1],
                              'Pressão do Ar': [32.1],
                              'Pressão ATM': [1008.20],
                              'Umidade do Ar':[70.99]
})

print("R2: ",modelo.rsquared)
print("Valores de Energia Obtida: ", modelo.predict(novo_dado)[0])