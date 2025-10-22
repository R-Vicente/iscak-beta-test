# Instruções para Gerar Datasets

## 📋 Passo a Passo

### 1. Instalar dependências

```bash
pip install scikit-learn pandas seaborn ucimlrepo
```

### 2. Executar o script

```bash
python generate_datasets.py
```

### 3. Verificar os arquivos gerados

Deve aparecer uma pasta `datasets/` com 7 arquivos:

```
datasets/
├── iris.json          (Pure Numeric)
├── wine.json          (Pure Numeric)
├── diabetes.json      (Pure Numeric)
├── sonar.json         (Pure Numeric)
├── titanic.json       (Mixed)
├── credit.json        (Mixed)
└── mushroom.json      (Pure Categorical)
```

### 4. Fazer commit

```bash
git add datasets/
git add generate_datasets.py
git add DATASETS_README.md
git commit -m "Add datasets for Test Methods page"
git push
```

## ⚠️ Possíveis problemas

**Problema**: Erro ao baixar Credit ou Mushroom (UCI)
**Solução**: Verifica a conexão internet e tenta novamente

**Problema**: `ModuleNotFoundError: No module named 'ucimlrepo'`
**Solução**: `pip install ucimlrepo`

## 📊 Datasets gerados

| Dataset | Tipo | Rows | Cols | Fonte |
|---------|------|------|------|-------|
| IRIS | Pure Numeric | 150 | 4 | sklearn |
| WINE | Pure Numeric | 178 | 13 | sklearn |
| Diabetes | Pure Numeric | 442 | 10 | sklearn |
| SONAR | Pure Numeric | 208 | 60 | OpenML |
| Titanic | Mixed | ~600 | 10 | Seaborn |
| Credit | Mixed | ~600 | 15 | UCI |
| Mushroom | Pure Categorical | ~8000 | 22 | UCI |

## ✅ Formato JSON

Cada arquivo JSON tem esta estrutura:

```json
{
  "name": "IRIS",
  "type": "Pure Numeric",
  "rows": 150,
  "columns": 4,
  "column_names": ["sepal length", "sepal width", ...],
  "data": [[5.1, 3.5, 1.4, 0.2], ...]
}
```

Para datasets **Mixed**, também tem:
```json
{
  ...
  "categorical_columns": ["sex", "embarked"],
  "numeric_columns": ["age", "fare"]
}
```
