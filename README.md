# 🔥 OrbitFire AI — Previsão de Risco de Queimadas

**GAIE — Generative AI for Engineering | FIAP Global Solution**
**Squad: Sentinela Corp** · [github.com/sentinela-corp/sentinela-app-gaie](https://github.com/sentinela-corp/sentinela-app-gaie)

> Sistema de Machine Learning para classificação de risco de queimadas (baixo / médio / alto) com base em variáveis climáticas e satelitais, com interpretabilidade via SHAP e deploy em Streamlit.

**ODS 13 — Ação Climática** 🌍

---

## 📁 Estrutura do Repositório

```
orbitfire-ai-gaie/
│
├── modelo_risco_queimada.py   # Pipeline completo: dados → treino → avaliação → export
├── app_streamlit.py           # Dashboard interativo Streamlit
├── shap_analysis.ipynb        # Notebook de interpretabilidade SHAP
├── requirements.txt           # Dependências Python
├── README.md                  # Este arquivo
│
└── outputs/                   # Gerado ao executar o pipeline
    ├── dataset_orbitfire.csv
    ├── modelo_orbitfire.pkl
    ├── model_metadata.json
    ├── confusion_matrices.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── shap_*.png
```

---

## 🚀 Como Executar

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/sentinela-corp/sentinela-app-gaie.git
cd sentinela-app-gaie

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Treinar os modelos

```bash
python modelo_risco_queimada.py
```

O script executa automaticamente:
- ✅ Geração do dataset sintético (3.000 amostras)
- ✅ Pré-processamento e engenharia de atributos (14 features)
- ✅ Treinamento de **2 modelos**: Random Forest e Gradient Boosting
- ✅ Validação cruzada (5-fold) e comparação de métricas
- ✅ Exportação do melhor modelo (`outputs/modelo_orbitfire.pkl`)
- ✅ Geração de gráficos de avaliação

### 3. Iniciar o dashboard

```bash
streamlit run app_streamlit.py
```

Acesse em `http://localhost:8501`

### 4. Executar análise SHAP (notebook)

```bash
jupyter notebook shap_analysis.ipynb
```

---

## 🧠 Solução de ML

### Variáveis de entrada (9 originais → 14 após feature engineering)

| Feature | Descrição |
|---|---|
| `temperatura` | Temperatura do ar em °C |
| `umidade_relativa` | Umidade relativa do ar em % |
| `velocidade_vento` | Velocidade do vento em km/h |
| `precipitacao_24h` | Precipitação nas últimas 24h em mm |
| `ndvi` | Índice de vegetação (−0.1 a 0.9) |
| `dias_sem_chuva` | Dias consecutivos sem chuva |
| `altitude` | Altitude em metros |
| `mes` | Mês do ano (1–12) |
| `historico_focos` | Focos de incêndio nos últimos 30 dias |
| `indice_aridez` ✨ | temperatura / umidade (derivada) |
| `potencial_propagacao` ✨ | vento × (1 − umidade) (derivada) |
| `deficit_hidrico` ✨ | dias_sem_chuva × (1 − chuva recente) (derivada) |
| `estacao_seca` ✨ | 1 se maio–outubro, 0 caso contrário |
| `ndvi_temp` ✨ | Interação NDVI × temperatura |

### Modelos treinados

| Modelo | Configuração principal |
|---|---|
| **Random Forest** | 200 árvores, max_depth=12, class_weight=balanced |
| **Gradient Boosting** | 200 estimadores, lr=0.05, max_depth=5, subsample=0.8 |

### Métricas de avaliação
- Acurácia no conjunto de teste
- F1-Score ponderado (weighted)
- Validação cruzada (5-fold, StratifiedKFold)
- Matriz de confusão por classe
- Feature Importance (Random Forest)

### Interpretabilidade — SHAP
- Summary Plot (importância global)
- Beeswarm Plot (distribuição de impactos)
- Waterfall Plot (explicação individual)
- Análise por classe de risco
- Gráfico de Dependência Parcial

---

## 📊 Resultados Esperados

Os modelos atingem tipicamente:

| Modelo | Acurácia | F1-Score |
|---|---|---|
| Random Forest | ~88–92% | ~88–92% |
| Gradient Boosting | ~86–90% | ~86–90% |

> Resultados variam conforme o seed. Random Forest tende a ser mais estável.

---

## 🖥️ Dashboard Streamlit

O app oferece:
- **Predição em tempo real** com sliders para todos os parâmetros
- **Probabilidades** para cada classe (baixo / médio / alto)
- **Análise SHAP** individual explicando a predição atual
- **Comparação de modelos** com métricas e gráficos
- **Mapa de risco simulado** (fronteira de decisão temperatura × umidade)

---

## 🌍 Conexão com ODS

| ODS | Contribuição |
|---|---|
| **ODS 13** — Ação Climática | Detecção precoce de queimadas reduz emissão de CO₂ e impactos climáticos |
| **ODS 15** — Vida Terrestre | Proteção de biomas como Cerrado e Amazônia |
| **ODS 11** — Cidades Sustentáveis | Alertas para populações em áreas de risco |

---

## 👥 Equipe — Sentinela Corp

| Nome | RM |
|---|---|
| Deivison Pertel | RM 550803 |
| Eduardo Akira Murata | RM 98713 |
| Wesley Souza de Oliveira | RM 97874 |

🔗 **GitHub:** [github.com/sentinela-corp/sentinela-app-gaie](https://github.com/sentinela-corp/sentinela-app-gaie)

**Turma:** 4ESR | **Professor(a):** Francisco Elanio Bezerra | **Ano:** 2026

---

## 📚 Referências

- INPE — [BDQueimadas](http://queimadas.dgi.inpe.br/)
- NASA FIRMS — [Fire Information for Resource Management](https://firms.modaps.eosdis.nasa.gov/)
- Canadian Forest Service — [Fire Weather Index (FWI)](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)
- Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
- Scikit-learn Documentation — [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

---

*Desenvolvido pela **Sentinela Corp** para a Global Solution FIAP — GAIE (Generative AI for Engineering)*
*[github.com/sentinela-corp/sentinela-app-gaie](https://github.com/sentinela-corp/sentinela-app-gaie)*
