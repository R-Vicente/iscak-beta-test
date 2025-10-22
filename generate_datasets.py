"""
Script para gerar datasets em formato JSON para a página Test Methods.

Dependências necessárias:
pip install scikit-learn pandas seaborn ucimlrepo

Datasets gerados:
- Pure Numeric: IRIS, WINE, Diabetes, SONAR
- Mixed: Titanic, Credit Approval
- Pure Categorical: Mushroom
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_openml
import seaborn as sns
from ucimlrepo import fetch_ucirepo


def create_datasets_folder():
    """Cria a pasta datasets se não existir."""
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
        print("✓ Pasta 'datasets' criada")
    else:
        print("✓ Pasta 'datasets' já existe")


def save_dataset_json(data_dict, filename, dataset_name):
    """Salva dataset em formato JSON."""
    filepath = os.path.join('datasets', filename)

    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print(f"✓ {dataset_name}: {filepath} ({data_dict['rows']} rows, {data_dict['columns']} cols)")


def generate_iris():
    """Dataset IRIS - Pure Numeric."""
    print("\n[1/7] Gerando IRIS...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    dataset = {
        'name': 'IRIS',
        'type': 'Pure Numeric',
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data': df.values.tolist()
    }

    save_dataset_json(dataset, 'iris.json', 'IRIS')


def generate_wine():
    """Dataset WINE - Pure Numeric."""
    print("\n[2/7] Gerando WINE...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    dataset = {
        'name': 'WINE',
        'type': 'Pure Numeric',
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data': df.values.tolist()
    }

    save_dataset_json(dataset, 'wine.json', 'WINE')


def generate_diabetes():
    """Dataset Diabetes - Pure Numeric."""
    print("\n[3/7] Gerando Diabetes...")
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    dataset = {
        'name': 'Diabetes',
        'type': 'Pure Numeric',
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data': df.values.tolist()
    }

    save_dataset_json(dataset, 'diabetes.json', 'Diabetes')


def generate_sonar():
    """Dataset SONAR - Pure Numeric."""
    print("\n[4/7] Gerando SONAR...")
    sonar_data = fetch_openml('sonar', version=1, parser='auto', as_frame=True)
    df = sonar_data.data

    # Converter todas as colunas para numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    dataset = {
        'name': 'SONAR',
        'type': 'Pure Numeric',
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data': df.values.tolist()
    }

    save_dataset_json(dataset, 'sonar.json', 'SONAR')


def generate_titanic():
    """Dataset Titanic - Mixed."""
    print("\n[5/7] Gerando Titanic...")
    df = sns.load_dataset('titanic')

    # Remover colunas com muitos missings nativos e selecionar features relevantes
    df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who', 'adult_male', 'alone']]

    # Remover linhas com missings nativos
    df = df.dropna()

    # Identificar colunas categóricas e numéricas
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    dataset = {
        'name': 'Titanic',
        'type': 'Mixed',
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'categorical_columns': categorical_cols,
        'numeric_columns': numeric_cols,
        'data': df.values.tolist()
    }

    save_dataset_json(dataset, 'titanic.json', 'Titanic')


def generate_credit():
    """Dataset Credit Approval - Mixed."""
    print("\n[6/7] Gerando Credit Approval...")

    try:
        credit_approval = fetch_ucirepo(id=27)
        X = credit_approval.data.features

        # Remover linhas com missings nativos
        X = X.dropna()

        # Identificar colunas categóricas e numéricas
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

        dataset = {
            'name': 'Credit Approval',
            'type': 'Mixed',
            'rows': len(X),
            'columns': len(X.columns),
            'column_names': X.columns.tolist(),
            'categorical_columns': categorical_cols,
            'numeric_columns': numeric_cols,
            'data': X.values.tolist()
        }

        save_dataset_json(dataset, 'credit.json', 'Credit Approval')

    except Exception as e:
        print(f"✗ Erro ao carregar Credit Approval: {e}")
        print("  Tenta novamente ou verifica a conexão")


def generate_mushroom():
    """Dataset Mushroom - Pure Categorical."""
    print("\n[7/7] Gerando Mushroom...")

    try:
        mushroom = fetch_ucirepo(id=73)
        X = mushroom.data.features

        # Remover linhas com missings nativos
        X = X.dropna()

        dataset = {
            'name': 'Mushroom',
            'type': 'Pure Categorical',
            'rows': len(X),
            'columns': len(X.columns),
            'column_names': X.columns.tolist(),
            'data': X.values.tolist()
        }

        save_dataset_json(dataset, 'mushroom.json', 'Mushroom')

    except Exception as e:
        print(f"✗ Erro ao carregar Mushroom: {e}")
        print("  Tenta novamente ou verifica a conexão")


def main():
    """Função principal para gerar todos os datasets."""
    print("=" * 60)
    print("GERADOR DE DATASETS PARA TEST METHODS")
    print("=" * 60)

    create_datasets_folder()

    # Gerar datasets
    generate_iris()
    generate_wine()
    generate_diabetes()
    generate_sonar()
    generate_titanic()
    generate_credit()
    generate_mushroom()

    print("\n" + "=" * 60)
    print("✓ CONCLUÍDO! Todos os datasets foram gerados.")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. git add datasets/")
    print("2. git commit -m 'Add datasets for Test Methods page'")
    print("3. git push")
    print("\n")


if __name__ == "__main__":
    main()
