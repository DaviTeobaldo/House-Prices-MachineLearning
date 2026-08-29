# House Prices - Advanced Regression Techniques

Projeto de Machine Learning desenvolvido como aplicação prática de técnicas de
regressão estudadas em disciplina de Aprendizado de Máquina, resolvendo a
competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
do Kaggle.

## Sobre o problema

O conjunto de dados utilizado é o *Ames Housing Dataset* (De Cock, 2011), que
descreve **79 variáveis explicativas** sobre imóveis residenciais em Ames,
Iowa (EUA). O objetivo é prever `SalePrice`, o preço de venda de cada imóvel,
a partir dessas variáveis.

A métrica oficial de avaliação é o **RMSLE (Root Mean Squared Logarithmic
Error)**, calculado entre o logaritmo do valor previsto e o logaritmo do
valor observado.

## Metodologia

O notebook segue um pipeline estruturado de ponta a ponta:

1. **Análise Exploratória de Dados (EDA)** — distribuição do alvo, padrão de
   valores ausentes, correlações e identificação de outliers;
2. **Pré-processamento** — tratamento de valores ausentes por significado
   semântico, engenharia de atributos, codificação ordinal e one-hot,
   correção de assimetria (`log1p`);
3. **Modelagem** — comparação de modelos lineares regularizados (Ridge,
   Lasso, ElasticNet) e modelos de ensemble baseados em árvores (Random
   Forest, Gradient Boosting), avaliados via validação cruzada K-Fold;
4. **Otimização de hiperparâmetros** com `GridSearchCV`;
5. **Ensemble** de modelos por combinação ponderada;
6. **Geração da submissão final** (`submission.csv`).

## Resultados

| Modelo | RMSLE médio (validação cruzada, 5-fold) |
|---|---|
| Regressão Linear | 0.2249 |
| Random Forest | 0.1365 |
| Gradient Boosting | 0.1242 |
| Lasso | 0.1155 |
| ElasticNet | 0.1154 |
| **Ridge** | **0.1145** |

O modelo final de submissão combina Ridge e Gradient Boosting em um ensemble
ponderado, validado por hold-out.

## Tecnologias

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- scipy

## Estrutura do repositório

```
.
├── House_Prices_ML_Colab.ipynb   # Notebook principal (Google Colab)
├── train.csv                     # Dados de treino (Kaggle)
├── test.csv                      # Dados de teste (Kaggle)
└── README.md
```

## Como executar

1. Baixe `train.csv` e `test.csv` na aba [Data da competição no Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2. Abra `House_Prices_ML_Colab.ipynb` no [Google Colab](https://colab.research.google.com/).
3. Execute as células em ordem; quando solicitado, faça o upload dos dois arquivos `.csv`.
4. Ao final, o arquivo `submission.csv` é gerado e disponibilizado para download.

## Autor

Davi — Instituto de Matemática e Estatística, Universidade de São Paulo (IME-USP).

## Licença

Uso acadêmico. Dados de propriedade da competição Kaggle / Ames Housing Dataset.
