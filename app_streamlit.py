"""
OrbitFire AI — Dashboard Streamlit
GAIE - Generative AI for Engineering | Global Solution

Squad: Sentinela Corp — https://github.com/sentinela-corp/sentinela-app-gaie
Autores:
  - Deivison Pertel          RM 550803
  - Eduardo Akira Murata     RM 98713
  - Wesley Souza de Oliveira RM 97874

Interface web para previsão interativa de risco de queimadas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os
import shap
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="OrbitFire AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E25822;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 8px;
    }
    .risk-alto   { background: #FFE5E5; color: #C0392B; border: 2px solid #C0392B; }
    .risk-médio  { background: #FFF8E1; color: #E67E22; border: 2px solid #E67E22; }
    .risk-baixo  { background: #E8F8F0; color: #27AE60; border: 2px solid #27AE60; }
    .metric-box {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

@st.cache_resource
def carregar_modelo():
    """Carrega modelo treinado e metadados."""
    modelo_path = "outputs/modelo_orbitfire.pkl"
    meta_path   = "outputs/model_metadata.json"

    if not os.path.exists(modelo_path):
        st.error("⚠️ Modelo não encontrado! Execute `python modelo_risco_queimada.py` primeiro.")
        st.stop()

    modelo = joblib.load(modelo_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return modelo, meta


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replica a engenharia de atributos do treino."""
    df = df.copy()
    df["indice_aridez"]        = df["temperatura"] / (df["umidade_relativa"] + 1)
    df["potencial_propagacao"] = df["velocidade_vento"] * (1 - df["umidade_relativa"] / 100)
    df["deficit_hidrico"]      = df["dias_sem_chuva"] * (1 - np.clip(df["precipitacao_24h"] / 10, 0, 1))
    df["estacao_seca"]         = df["mes"].apply(lambda m: 1 if 5 <= m <= 10 else 0)
    df["ndvi_temp"]            = df["ndvi"] * df["temperatura"]
    return df


CORES_RISCO = {"alto": "#C0392B", "médio": "#E67E22", "baixo": "#27AE60"}
ICONES_RISCO = {"alto": "🔴", "médio": "🟡", "baixo": "🟢"}


# ─────────────────────────────────────────────
# CARREGAR MODELO
# ─────────────────────────────────────────────

modelo, meta = carregar_modelo()
features = meta["features"]


# ─────────────────────────────────────────────
# SIDEBAR — ENTRADAS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛰️ Parâmetros Climáticos")
    st.markdown("Ajuste os valores para prever o risco.")
    st.caption("**Sentinela Corp** · [GitHub](https://github.com/sentinela-corp/sentinela-app-gaie)")
    st.caption("Deivison Pertel · Eduardo Akira Murata · Wesley Souza de Oliveira")

    temperatura      = st.slider("🌡️ Temperatura (°C)",       10.0, 45.0, 32.0, 0.5)
    umidade_rel      = st.slider("💧 Umidade Relativa (%)",    5.0,  95.0, 25.0, 1.0)
    velocidade_vento = st.slider("🌬️ Vento (km/h)",            0.0,  80.0, 35.0, 1.0)
    precipitacao_24h = st.slider("🌧️ Precipitação 24h (mm)",   0.0,  50.0,  0.5, 0.5)
    ndvi             = st.slider("🌿 NDVI (Vegetação)",        -0.1,  0.9,  0.3, 0.01)
    dias_sem_chuva   = st.slider("☀️ Dias sem chuva",           0,   120,   45)
    altitude         = st.slider("⛰️ Altitude (m)",             0,  2500,  600)
    mes              = st.selectbox("📅 Mês", options=list(range(1, 13)),
                                    format_func=lambda m: [
                                        "", "Jan","Fev","Mar","Abr","Mai","Jun",
                                        "Jul","Ago","Set","Out","Nov","Dez"][m],
                                    index=6)
    historico_focos  = st.slider("🔥 Focos últimos 30 dias",   0,   30,    8)

    st.divider()
    st.markdown(f"**Melhor modelo:** {meta['melhor_modelo']}")
    for nome, m in meta["metricas"].items():
        st.caption(f"{nome}: F1={m['f1_score']:.3f} | Acc={m['accuracy']:.3f}")


# ─────────────────────────────────────────────
# PAINEL PRINCIPAL
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">🔥 OrbitFire AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistema de Previsão de Risco de Queimadas por Dados Satelitais · FIAP Global Solution</div>', unsafe_allow_html=True)

# ── Previsão ──
entrada = pd.DataFrame([{
    "temperatura":      temperatura,
    "umidade_relativa": umidade_rel,
    "velocidade_vento": velocidade_vento,
    "precipitacao_24h": precipitacao_24h,
    "ndvi":             ndvi,
    "dias_sem_chuva":   dias_sem_chuva,
    "altitude":         altitude,
    "mes":              mes,
    "historico_focos":  historico_focos,
}])

entrada_eng = engenharia_features(entrada)
entrada_eng = entrada_eng[features]

risco_pred  = modelo.predict(entrada_eng)[0]
proba       = modelo.predict_proba(entrada_eng)[0]
classes     = modelo.classes_

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"""
    <div class="risk-card risk-{risco_pred}">
        {ICONES_RISCO[risco_pred]} Risco <strong>{risco_pred.upper()}</strong> de Queimada
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("**Probabilidades**")
    for cls, prob in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
        st.metric(label=f"{ICONES_RISCO[cls]} {cls.capitalize()}", value=f"{prob:.1%}")

with col3:
    st.markdown("**Features derivadas**")
    ia = round(temperatura / (umidade_rel + 1), 2)
    pp = round(velocidade_vento * (1 - umidade_rel / 100), 2)
    dh = round(dias_sem_chuva * (1 - min(precipitacao_24h / 10, 1)), 2)
    st.metric("Índice Aridez",          ia)
    st.metric("Potencial Propagação",   pp)
    st.metric("Déficit Hídrico",        dh)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Análise SHAP", "📈 Métricas dos Modelos", "🗺️ Mapa de Risco Simulado"])

with tab1:
    st.subheader("Interpretabilidade — SHAP")
    st.caption("Entenda quais variáveis mais influenciaram a predição atual.")

    try:
        with st.spinner("Calculando SHAP..."):
            inner_model = modelo.named_steps["model"]
            scaler      = modelo.named_steps["scaler"]
            X_scaled    = scaler.transform(entrada_eng)

            explainer   = shap.TreeExplainer(inner_model)
            shap_vals   = explainer.shap_values(X_scaled)

            # Para multiclasse, pegar a classe predita
            idx_classe  = list(classes).index(risco_pred)

            if isinstance(shap_vals, list):
                sv = shap_vals[idx_classe][0]
            else:
                sv = shap_vals[0]

            fig, ax = plt.subplots(figsize=(9, 5))
            sv_series = pd.Series(sv, index=features).sort_values()
            cores_bar = ["#C0392B" if v > 0 else "#27AE60" for v in sv_series]
            sv_series.plot(kind="barh", ax=ax, color=cores_bar, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"SHAP — Explicação para risco '{risco_pred}'", fontweight="bold")
            ax.set_xlabel("Contribuição SHAP")
            ax.grid(axis="x", alpha=0.3)
            red_patch   = mpatches.Patch(color="#C0392B", label="Aumenta risco")
            green_patch = mpatches.Patch(color="#27AE60", label="Reduz risco")
            ax.legend(handles=[red_patch, green_patch])
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP indisponível para este modelo: {e}")

with tab2:
    st.subheader("Comparação de Modelos")
    metricas_df = pd.DataFrame(meta["metricas"]).T[["accuracy", "f1_score", "cv_mean", "cv_std"]]
    metricas_df.columns = ["Acurácia", "F1-Score", "CV Média", "CV Desvio"]
    st.dataframe(metricas_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)

    # Gráfico de barras de métricas
    if os.path.exists("outputs/model_comparison.png"):
        st.image("outputs/model_comparison.png")

    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png")

with tab3:
    st.subheader("Simulação de Mapa de Risco — Grid Amostral")
    st.caption("Variando temperatura e umidade para visualizar a fronteira de decisão.")

    n_pts = 30
    temps  = np.linspace(10, 45, n_pts)
    umids  = np.linspace(5, 95, n_pts)
    TT, UU = np.meshgrid(temps, umids)

    grid_rows = []
    for t, u in zip(TT.ravel(), UU.ravel()):
        grid_rows.append({
            "temperatura": t, "umidade_relativa": u,
            "velocidade_vento": velocidade_vento,
            "precipitacao_24h": precipitacao_24h,
            "ndvi": ndvi, "dias_sem_chuva": dias_sem_chuva,
            "altitude": altitude, "mes": mes,
            "historico_focos": historico_focos,
        })

    grid_df  = engenharia_features(pd.DataFrame(grid_rows))[features]
    preds    = modelo.predict(grid_df)
    mapa_int = np.array([{"baixo": 0, "médio": 1, "alto": 2}[p] for p in preds])
    ZZ       = mapa_int.reshape(n_pts, n_pts)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.colors.ListedColormap(["#27AE60", "#E67E22", "#C0392B"])
    contour = ax.contourf(TT, UU, ZZ, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.75)
    ax.set_xlabel("Temperatura (°C)", fontweight="bold")
    ax.set_ylabel("Umidade Relativa (%)", fontweight="bold")
    ax.set_title("Fronteira de Decisão — Risco de Queimada", fontweight="bold")

    # Ponto atual
    ax.scatter([temperatura], [umidade_rel], c="black", s=120, zorder=5,
               marker="*", label=f"Ponto atual → {risco_pred}")
    ax.legend()

    patches = [
        mpatches.Patch(color="#27AE60", label="Baixo"),
        mpatches.Patch(color="#E67E22", label="Médio"),
        mpatches.Patch(color="#C0392B", label="Alto"),
    ]
    ax.legend(handles=patches + [ax.get_legend_handles_labels()[0][-1]],
              labels=["Baixo", "Médio", "Alto", f"Ponto atual ({temperatura}°C, {umidade_rel}%)"])
    st.pyplot(fig)


# ─────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────

st.divider()
st.caption("🛰️ OrbitFire AI · Sentinela Corp · github.com/sentinela-corp/sentinela-app-gaie · GAIE — Generative AI for Engineering | ODS 13 — Ação Climática")
