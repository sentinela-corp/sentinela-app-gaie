"""
OrbitFire AI — Modelo de Previsão de Risco de Queimadas
GAIE - Generative AI for Engineering | Global Solution

Squad: Sentinela Corp — https://github.com/sentinela-corp/sentinela-app-gaie
Autores:
  - Deivison Pertel          RM 550803
  - Eduardo Akira Murata     RM 98713
  - Wesley Souza de Oliveira RM 97874

Treina e compara dois modelos de ML para classificar risco de queimada
(baixo, médio, alto) com base em variáveis climáticas e ambientais.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GERAÇÃO / COLETA DE DADOS SINTÉTICOS
# ─────────────────────────────────────────────

def gerar_dados(n_amostras: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Gera dataset sintético baseado em padrões climáticos reais de queimadas.
    Variáveis baseadas nos índices FWI (Fire Weather Index) e estudos do INPE.
    """
    np.random.seed(seed)

    temperatura   = np.random.uniform(10, 45, n_amostras)        # °C
    umidade_rel   = np.random.uniform(5, 95, n_amostras)          # %
    velocidade_vento = np.random.uniform(0, 80, n_amostras)       # km/h
    precipitacao  = np.random.exponential(5, n_amostras)          # mm (últimas 24h)
    ndvi          = np.random.uniform(-0.1, 0.9, n_amostras)      # índice vegetação
    dias_sem_chuva = np.random.randint(0, 120, n_amostras)        # dias
    altitude      = np.random.uniform(0, 2500, n_amostras)        # metros
    mes           = np.random.randint(1, 13, n_amostras)          # 1–12
    historico_local = np.random.poisson(3, n_amostras)            # focos últimos 30d

    # Sazonalidade: meses de seca (maio–outubro no cerrado/amazônia)
    sazonalidade = np.where((mes >= 5) & (mes <= 10), 1.4, 0.7)

    # Score de risco composto (lógica baseada em especialistas)
    score = (
          0.30 * (temperatura / 45)
        - 0.25 * (umidade_rel / 100)
        + 0.15 * (velocidade_vento / 80)
        - 0.10 * np.clip(precipitacao / 20, 0, 1)
        + 0.10 * np.clip(dias_sem_chuva / 60, 0, 1)
        - 0.05 * ndvi
        + 0.05 * (historico_local / 15)
    ) * sazonalidade

    # Adicionar ruído
    score += np.random.normal(0, 0.05, n_amostras)
    score = np.clip(score, 0, 1)

    # Classificação em 3 faixas (tercis para balanceamento)
    risco = pd.cut(
        score,
        bins=np.percentile(score, [0, 33, 66, 100]),
        labels=["baixo", "médio", "alto"],
        include_lowest=True
    )

    df = pd.DataFrame({
        "temperatura":       temperatura,
        "umidade_relativa":  umidade_rel,
        "velocidade_vento":  velocidade_vento,
        "precipitacao_24h":  precipitacao,
        "ndvi":              ndvi,
        "dias_sem_chuva":    dias_sem_chuva,
        "altitude":          altitude,
        "mes":               mes,
        "historico_focos":   historico_local,
        "risco_queimada":    risco
    })

    return df


# ─────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO E ENGENHARIA DE ATRIBUTOS
# ─────────────────────────────────────────────

def preprocessar(df: pd.DataFrame):
    """Limpeza, feature engineering e split treino/teste."""

    # Remoção de nulos (caso existam em dados reais)
    df = df.dropna().copy()

    # ── Engenharia de atributos ──
    # Índice de aridez simplificado
    df["indice_aridez"] = df["temperatura"] / (df["umidade_relativa"] + 1)

    # Potencial de propagação do fogo
    df["potencial_propagacao"] = df["velocidade_vento"] * (1 - df["umidade_relativa"] / 100)

    # Déficit hídrico
    df["deficit_hidrico"] = df["dias_sem_chuva"] * (1 - np.clip(df["precipitacao_24h"] / 10, 0, 1))

    # Flag estação seca
    df["estacao_seca"] = df["mes"].apply(lambda m: 1 if 5 <= m <= 10 else 0)

    # Interação NDVI × temperatura
    df["ndvi_temp"] = df["ndvi"] * df["temperatura"]

    features = [
        "temperatura", "umidade_relativa", "velocidade_vento",
        "precipitacao_24h", "ndvi", "dias_sem_chuva", "altitude",
        "mes", "historico_focos",
        "indice_aridez", "potencial_propagacao", "deficit_hidrico",
        "estacao_seca", "ndvi_temp"
    ]

    X = df[features]
    y = df["risco_queimada"].astype(str)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), features


# ─────────────────────────────────────────────
# 3. TREINAMENTO DE MODELOS
# ─────────────────────────────────────────────

def treinar_modelos(X_train, y_train):
    """Treina Random Forest e Gradient Boosting com pipelines."""

    scaler = StandardScaler()

    # Modelo 1 — Random Forest
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Modelo 2 — Gradient Boosting
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])

    print("🌲 Treinando Random Forest...")
    rf.fit(X_train, y_train)

    print("🚀 Treinando Gradient Boosting...")
    gb.fit(X_train, y_train)

    return {"Random Forest": rf, "Gradient Boosting": gb}


# ─────────────────────────────────────────────
# 4. VALIDAÇÃO E COMPARAÇÃO
# ─────────────────────────────────────────────

def avaliar_modelos(modelos: dict, X_train, X_test, y_train, y_test) -> dict:
    """Avalia todos os modelos e retorna métricas consolidadas."""

    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("📊  AVALIAÇÃO DOS MODELOS")
    print("=" * 60)

    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)

        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="weighted")
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)

        print(f"\n── {nome} ──")
        print(f"  Acurácia (teste):    {acc:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        print(f"  CV Acurácia:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\n" + classification_report(y_test, y_pred))

        resultados[nome] = {
            "accuracy": acc,
            "f1_score": f1,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "y_pred":   y_pred
        }

    return resultados


# ─────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ─────────────────────────────────────────────

def gerar_visualizacoes(modelos, resultados, X_test, y_test, feature_names, output_dir="outputs"):
    """Salva gráficos de matriz de confusão e comparação de métricas."""
    os.makedirs(output_dir, exist_ok=True)

    classes = ["alto", "baixo", "médio"]
    cores   = {"Random Forest": "#E25822", "Gradient Boosting": "#2C6FAC"}

    # ── Matrizes de Confusão ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OrbitFire AI — Matrizes de Confusão", fontsize=14, fontweight="bold")

    for ax, (nome, res) in zip(axes, resultados.items()):
        cm = confusion_matrix(y_test, res["y_pred"], labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(nome, fontweight="bold", color=cores[nome])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_dir}/confusion_matrices.png")

    # ── Comparação de Métricas ──
    metricas = ["accuracy", "f1_score", "cv_mean"]
    labels   = ["Acurácia", "F1-Score", "CV Acurácia"]
    x = np.arange(len(metricas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (nome, res) in enumerate(resultados.items()):
        vals = [res[m] for m in metricas]
        bars = ax.bar(x + i * width, vals, width, label=nome,
                      color=list(cores.values())[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_title("Comparação de Modelos — OrbitFire AI", fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_dir}/model_comparison.png")

    # ── Feature Importance (Random Forest) ──
    rf_model = modelos["Random Forest"].named_steps["model"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot(kind="barh", ax=ax, color="#E25822", alpha=0.85, edgecolor="white")
    ax.set_title("Importância de Variáveis — Random Forest", fontweight="bold")
    ax.set_xlabel("Importância")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_dir}/feature_importance.png")


# ─────────────────────────────────────────────
# 6. SALVAR MODELO E METADADOS
# ─────────────────────────────────────────────

def salvar_modelo(modelos, resultados, feature_names, output_dir="outputs"):
    """Persiste o melhor modelo e metadados para uso no Streamlit."""
    os.makedirs(output_dir, exist_ok=True)

    # Selecionar melhor modelo por F1-Score
    melhor_nome = max(resultados, key=lambda k: resultados[k]["f1_score"])
    melhor_modelo = modelos[melhor_nome]

    joblib.dump(melhor_modelo, f"{output_dir}/modelo_orbitfire.pkl")
    print(f"\n🏆 Melhor modelo: {melhor_nome} (F1={resultados[melhor_nome]['f1_score']:.4f})")
    print(f"✅ Modelo salvo em {output_dir}/modelo_orbitfire.pkl")

    # Salvar metadados
    meta = {
        "melhor_modelo":  melhor_nome,
        "features":       feature_names,
        "classes":        ["alto", "baixo", "médio"],
        "metricas": {
            nome: {k: float(v) for k, v in res.items() if k != "y_pred"}
            for nome, res in resultados.items()
        },
        "gerado_em": datetime.now().isoformat()
    }
    with open(f"{output_dir}/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadados salvos em {output_dir}/model_metadata.json")

    return melhor_nome, melhor_modelo


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🔥 OrbitFire AI — Pipeline de Treinamento")
    print("=" * 60)

    # 1. Dados
    print("\n📡 Gerando dataset...")
    df = gerar_dados(n_amostras=3000)
    print(f"   {len(df)} amostras | distribuição:\n{df['risco_queimada'].value_counts()}")
    df.to_csv("outputs/dataset_orbitfire.csv", index=False)

    # 2. Pré-processamento
    print("\n⚙️  Pré-processando...")
    (X_train, X_test, y_train, y_test), features = preprocessar(df)
    print(f"   Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"   Features ({len(features)}): {features}")

    # 3. Treinamento
    print("\n🤖 Treinando modelos...")
    modelos = treinar_modelos(X_train, y_train)

    # 4. Avaliação
    resultados = avaliar_modelos(modelos, X_train, X_test, y_train, y_test)

    # 5. Visualizações
    print("\n🖼️  Gerando visualizações...")
    gerar_visualizacoes(modelos, resultados, X_test, y_test, features)

    # 6. Salvar
    salvar_modelo(modelos, resultados, features)

    print("\n✅ Pipeline concluído com sucesso!")
    print("   Sentinela Corp — https://github.com/sentinela-corp/sentinela-app-gaie")
