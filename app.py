"""
BIC-Guard AI — Plateforme agentique de contrôle, validation et reporting
pour Bureau d'Information sur le Crédit (Creditinfo Central Africa / CEMAC).

3 agents IA (Llama via Groq) pilotés par un orchestrateur central :
  Agent 1 : Validateur & Auditeur de données (règles déterministes + analyse LLM)
  Agent 2 : Assistant support déclarants (chat basé sur un référentiel éditable)
  Agent 3 : Reporting & Analytics exécutif (métriques de conformité + synthèse LLM)

Lancement : streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    from groq import Groq
except ImportError:  # l'app reste utilisable sans LLM
    Groq = None

# ───────────────────────── Charte graphique ─────────────────────────
RED, DARK_RED, BLACK, LIGHT = "#C8102E", "#8B0000", "#1A1A1A", "#F8F9FA"
RED_SCALE = [[0, "#FFFFFF"], [0.35, "#F4B6BF"], [0.7, RED], [1, DARK_RED]]

CSS = """
<style>
:root { --red:#C8102E; --dark-red:#8B0000; --ink:#1A1A1A; --bg:#F8F9FA; }
.stApp { background: var(--bg); color: var(--ink); }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
h1,h2,h3 { color: var(--ink); letter-spacing: -0.01em; }

.hero { background: linear-gradient(120deg, var(--ink) 0%, var(--dark-red) 62%, var(--red) 100%);
        color:#fff; padding: 1.6rem 2rem; border-radius: 14px; display:flex;
        align-items:center; gap:1.6rem; animation: rise .8s ease-out both; }
.hero .brand { font-size: 2rem; font-weight: 300; line-height:1; white-space:nowrap; }
.hero .brand b { font-weight: 800; }
.hero .brand small { display:block; font-size:.7rem; letter-spacing:.14em; opacity:.8; margin-top:.35rem; }
.hero h1 { color:#fff; font-size:1.7rem; margin:0; }
.hero p  { color:#ffd9de; margin:.25rem 0 0; font-size:.95rem; }
@keyframes rise { from { opacity:0; transform: translateY(14px);} to { opacity:1; transform:none;} }

.ticker { overflow:hidden; background:var(--ink); color:#fff; margin:.8rem 0 1.1rem;
          border-radius:8px; border-left:6px solid var(--red); }
.ticker-track { display:inline-block; white-space:nowrap; padding:.55rem 0;
                animation: scroll 38s linear infinite; }
.ticker:hover .ticker-track { animation-play-state: paused; }
.ticker span { margin: 0 2.2rem; font-size:.9rem; }
.ticker span::before { content:"\\25CF"; color:var(--red); margin-right:.6rem; }
@keyframes scroll { from { transform: translateX(0);} to { transform: translateX(-50%);} }

.kpi { background:#fff; border-radius:12px; padding:1rem 1.2rem; border-top:5px solid var(--red);
       box-shadow:0 2px 10px rgba(26,26,26,.07); }
.kpi .v { font-size:2.3rem; font-weight:800; color:var(--red); line-height:1.1; }
.kpi .l { font-size:.85rem; color:#555; margin-top:.2rem; }
.kpi.dark { border-top-color: var(--ink); } .kpi.dark .v { color: var(--ink); }

.stButton>button, .stDownloadButton>button { background:var(--red); color:#fff; border:0;
        border-radius:8px; font-weight:600; transition: background .2s, transform .15s; }
.stButton>button:hover, .stDownloadButton>button:hover { background:var(--dark-red); color:#fff; transform: translateY(-1px); }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stTabs [aria-selected="true"] { color: var(--red) !important; }
.agent-box { background:#fff; border-left:5px solid var(--red); border-radius:8px; padding:1rem 1.3rem; }
@media (prefers-reduced-motion: reduce) { .hero, .ticker-track { animation: none; } }
</style>
"""

# ───────────────────────── Données & référentiel ─────────────────────────
REQUIRED_COLS = ["declarant", "id_contrat", "id_client", "type_credit", "montant",
                 "encours", "taux", "date_octroi", "date_echeance", "statut",
                 "date_attendue", "date_transmission"]
DATE_COLS = ["date_octroi", "date_echeance", "date_attendue", "date_transmission"]
VALID_STATUTS = {"Normal", "Impayé", "Restructuré", "Contentieux", "Soldé"}
TAUX_MAX = 40.0  # % — seuil d'alerte (à aligner sur le taux d'usure en vigueur)

DEFAULT_REFERENTIEL = """\
(Référentiel illustratif — remplacez-le par les spécifications officielles de transmission Creditinfo.)
- Format : fichier CSV UTF-8, séparateur « ; » ou XLSX, une ligne par contrat.
- Champs obligatoires : declarant, id_contrat, id_client, type_credit, montant, encours, taux, date_octroi, date_echeance, statut, date_attendue, date_transmission.
- Dates au format AAAA-MM-JJ. Montants en XAF, sans séparateur de milliers. Taux en pourcentage annuel.
- id_contrat unique par déclarant. encours ≤ montant. date_echeance > date_octroi.
- Statuts admis : Normal, Impayé, Restructuré, Contentieux, Soldé.
- Transmission mensuelle attendue au plus tard le 10 du mois suivant la période.
- Un fichier avec erreurs critiques est retourné au déclarant pour correction et retransmission.
"""

DECLARANT_DELAY = {"Banque Alpha": -1, "Banque Beta": 2, "MFI Soleil": 6, "MFI Espoir": 9,
                   "Banque Gamma": 0, "Caisse Mutuelle Nord": 4, "Banque Delta": -2, "MFI Horizon": 12}


@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = ["Consommation", "Immobilier", "Équipement", "Trésorerie", "Microcrédit"]
    rows, k = [], 0
    for decl, mu in DECLARANT_DELAY.items():
        for per in pd.period_range("2026-04", "2026-09", freq="M"):
            attendue = per.end_time.normalize() + pd.Timedelta(days=10)
            transmis = attendue + pd.Timedelta(days=max(-4, int(rng.normal(mu, 3))))
            for _ in range(int(rng.integers(8, 16))):
                k += 1
                t = rng.choice(types)
                base = {"Microcrédit": 4e5, "Consommation": 2e6, "Trésorerie": 8e6,
                        "Équipement": 1.5e7, "Immobilier": 4e7}[t]
                montant = round(float(rng.lognormal(np.log(base), 0.35)), -3)
                octroi = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
                rows.append(dict(declarant=decl, id_contrat=f"{decl[:3].upper()}-{k:05d}",
                                 id_client=f"CL{rng.integers(10000, 99999)}", type_credit=t,
                                 montant=montant, encours=round(montant * float(rng.uniform(.1, 1)), -3),
                                 taux=round(float(rng.uniform(6, 22)), 2), date_octroi=octroi,
                                 date_echeance=octroi + pd.Timedelta(days=int(rng.integers(180, 2500))),
                                 statut=rng.choice(list(VALID_STATUTS), p=[.72, .1, .06, .04, .08]),
                                 date_attendue=attendue, date_transmission=transmis))
    df = pd.DataFrame(rows)
    # injection d'anomalies réalistes
    ix = rng.choice(df.index, 60, replace=False)
    df = pd.concat([df, df.loc[ix[:6]]], ignore_index=True)                  # doublons
    df.loc[ix[6:10], "montant"] = -df.loc[ix[6:10], "montant"]               # montants négatifs
    df.loc[ix[10:18], "encours"] = df.loc[ix[10:18], "montant"].abs() * 1.4  # encours > montant
    df.loc[ix[18:23], "date_echeance"] = df.loc[ix[18:23], "date_octroi"] - pd.Timedelta(days=30)
    df.loc[ix[23:27], "taux"] = 95.0                                         # taux aberrant
    df.loc[ix[27:31], "statut"] = "Inconnu"                                  # statut invalide
    df.loc[ix[31:37], "id_client"] = np.nan                                  # omissions
    df.loc[ix[37:42], "montant"] = df.loc[ix[37:42], "montant"].abs() * 25   # valeurs atypiques
    return df


def load_file(uploaded) -> tuple[pd.DataFrame, dict]:
    name = uploaded.name.lower()
    df = (pd.read_excel(uploaded) if name.endswith(("xlsx", "xls"))
          else pd.read_csv(uploaded, sep=None, engine="python"))
    df.columns = [c.strip().lower() for c in df.columns]
    bad_dates = {}
    for c in DATE_COLS:
        if c in df.columns:
            raw_missing = df[c].isna()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            bad_dates[c] = df.index[df[c].isna() & ~raw_missing].tolist()
    for c in ("montant", "encours", "taux"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, bad_dates


# ───────────────────────── Agent 1 : règles déterministes ─────────────────────────
SEV_W = {"Critique": 3, "Majeure": 2, "Mineure": 1}


def validate(df: pd.DataFrame, bad_dates: dict | None = None) -> pd.DataFrame:
    """Applique les règles BIC et retourne un journal d'anomalies (une ligne par anomalie)."""
    out = []
    has = lambda *c: all(x in df.columns for x in c)

    def add(mask, regle, cat, sev, detail):
        for i in df.index[mask.fillna(False)]:
            out.append(dict(ligne=int(i) + 2, declarant=df.at[i, "declarant"] if "declarant" in df else "?",
                            id_contrat=df.at[i, "id_contrat"] if "id_contrat" in df else "?",
                            regle=regle, categorie=cat, severite=sev, detail=detail))

    for c in REQUIRED_COLS:
        if c not in df.columns:
            out.append(dict(ligne=0, declarant="(fichier)", id_contrat="-", regle=f"Colonne manquante : {c}",
                            categorie="Structure", severite="Critique", detail="Colonne obligatoire absente"))
    for c in ("id_contrat", "id_client", "montant", "type_credit", "statut"):
        if c in df.columns:
            add(df[c].isna(), f"Valeur manquante : {c}", "Omission", "Majeure", f"{c} non renseigné")
    if has("declarant", "id_contrat"):
        add(df.duplicated(["declarant", "id_contrat"], keep="first"), "Doublon id_contrat", "Doublon",
            "Critique", "Contrat transmis plusieurs fois")
    if has("montant"):
        add(df["montant"] <= 0, "Montant ≤ 0", "Cohérence", "Critique", "Montant négatif ou nul")
    if has("montant", "encours"):
        add(df["encours"] > df["montant"], "Encours > montant", "Cohérence", "Majeure", "Encours supérieur au capital octroyé")
        add(df["encours"] < 0, "Encours négatif", "Cohérence", "Critique", "Encours négatif")
    if has("date_octroi", "date_echeance"):
        add(df["date_echeance"] <= df["date_octroi"], "Échéance ≤ octroi", "Cohérence", "Critique", "Chronologie incohérente")
    if has("taux"):
        add((df["taux"] > TAUX_MAX) | (df["taux"] < 0), f"Taux hors plage (0–{TAUX_MAX:.0f} %)", "Atypique",
            "Majeure", "Taux d'intérêt aberrant")
    if has("statut"):
        add(~df["statut"].isin(VALID_STATUTS) & df["statut"].notna(), "Statut invalide", "Format", "Majeure",
            f"Valeurs admises : {', '.join(sorted(VALID_STATUTS))}")
    if has("type_credit", "montant"):
        g = df.groupby("type_credit")["montant"]
        q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
        add(df["montant"] > q3 + 3 * (q3 - q1), "Montant atypique (IQR×3)", "Atypique", "Mineure",
            "Montant très supérieur à la distribution du type de crédit")
    for c, idx in (bad_dates or {}).items():
        for i in idx:
            out.append(dict(ligne=i + 2, declarant=df.at[i, "declarant"] if "declarant" in df else "?",
                            id_contrat=df.at[i, "id_contrat"] if "id_contrat" in df else "?",
                            regle=f"Date illisible : {c}", categorie="Format", severite="Majeure",
                            detail="Format attendu AAAA-MM-JJ"))
    cols = ["ligne", "declarant", "id_contrat", "regle", "categorie", "severite", "detail"]
    return pd.DataFrame(out, columns=cols)


def quality_score(n_rows: int, issues: pd.DataFrame) -> float:
    if n_rows == 0:
        return 0.0
    pen = issues["severite"].map(SEV_W).sum() / (n_rows * 3) * 100 * 4
    return float(max(0, round(100 - pen, 1)))


def batch_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["date_attendue", "date_transmission"]).copy()
    d["periode"] = d["date_attendue"].dt.to_period("M").sub(1).astype(str)  # période de référence
    d["retard_j"] = (d["date_transmission"] - d["date_attendue"]).dt.days
    b = d.groupby(["declarant", "periode"]).agg(retard_j=("retard_j", "median"), nb=("declarant", "size")).reset_index()
    b["a_temps"] = b["retard_j"] <= 0
    return b


# ───────────────────────── Orchestrateur & agents LLM ─────────────────────────
SYS_AUDIT = ("Tu es l'Agent Validateur & Auditeur d'un Bureau d'Information sur le Crédit (zone CEMAC). "
             "À partir UNIQUEMENT des résultats de contrôle fournis (JSON), rédige en français un rapport d'audit : "
             "1) Synthèse en 3 lignes, 2) Risques prioritaires classés, 3) Actions correctives par déclarant, "
             "4) Décision recommandée (Accepter / Accepter sous réserve / Retourner au déclarant) avec justification. "
             "N'invente aucun chiffre. Sois concis et précis.")
SYS_REPORT = ("Tu es l'Agent Reporting Exécutif d'un BIC (zone CEMAC). À partir UNIQUEMENT des métriques fournies (JSON), "
              "produis en français une note de décision pour la direction : constats clés, déclarants à risque, "
              "retards récurrents, tendance, recommandations d'action (relances, accompagnement, escalade). "
              "Maximum 250 mots, N'invente aucun chiffre.")


class Orchestrator:
    """Point d'entrée unique : route les tâches vers les agents et mutualise l'accès Groq/Llama."""

    def __init__(self, api_key: str, model: str, temperature: float):
        self.model, self.temperature = model, temperature
        self.client = Groq(api_key=api_key) if (Groq and api_key) else None

    @property
    def ready(self) -> bool:
        return self.client is not None

    def _call(self, messages, stream=False, max_tokens=1200):
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=max_tokens, stream=stream)

    def run(self, agent: str, **kw):
        return {"validateur": self.audit, "assistant": self.assist, "reporting": self.report}[agent](**kw)

    # Agent 1
    def audit(self, summary: dict) -> str:
        r = self._call([{"role": "system", "content": SYS_AUDIT},
                        {"role": "user", "content": json.dumps(summary, ensure_ascii=False, default=str)}])
        return r.choices[0].message.content

    # Agent 2 (flux)
    def assist(self, history: list, referentiel: str, context: dict):
        sys = ("Tu es l'Agent Assistant Support Déclarants d'un BIC. Tu aides banques et IMF à générer et transmettre "
               "des fichiers conformes. Réponds en français, de façon claire, avec des étapes concrètes. "
               "Appuie-toi sur le référentiel ci-dessous ; si l'information n'y figure pas, dis-le et propose de "
               "contacter l'équipe Creditinfo.\n\n=== RÉFÉRENTIEL ===\n" + referentiel +
               "\n\n=== DERNIER CONTRÔLE (contexte) ===\n" + json.dumps(context, ensure_ascii=False, default=str))
        stream = self._call([{"role": "system", "content": sys}] + history, stream=True, max_tokens=900)
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    # Agent 3
    def report(self, metrics: dict) -> str:
        r = self._call([{"role": "system", "content": SYS_REPORT},
                        {"role": "user", "content": json.dumps(metrics, ensure_ascii=False, default=str)}], max_tokens=900)
        return r.choices[0].message.content


# ───────────────────────── UI helpers ─────────────────────────
def style_fig(fig, h=360):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(color=BLACK), margin=dict(l=10, r=10, t=50, b=10),
                      colorway=[RED, DARK_RED, BLACK, "#E8707F", "#7A7A7A"])
    return fig


def kpi(col, value, label, dark=False):
    col.markdown(f'<div class="kpi {"dark" if dark else ""}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def llm_error(e: Exception):
    st.error(f"Appel Groq impossible : {e}\n\nVérifiez la clé API et le nom du modèle dans la barre latérale "
             "(les modèles disponibles évoluent : consultez console.groq.com/docs/models).")


# ───────────────────────── Application ─────────────────────────
st.set_page_config(page_title="BIC-Guard AI", page_icon="🛡️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ---- Barre latérale
with st.sidebar:
    st.header("Configuration")
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = st.text_input("Clé API Groq", value=secret_key, type="password",
                            help="Gratuite sur console.groq.com — ou via .streamlit/secrets.toml")
    model = st.text_input("Modèle", value="llama-3.3-70b-versatile",
                          help="Ex. llama-3.3-70b-versatile, llama-3.1-8b-instant, ou un modèle Llama 3.2 si votre compte y a accès.")
    temperature = st.slider("Température", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    st.subheader("Données")
    up = st.file_uploader("Fichier déclarant (CSV / XLSX)", type=["csv", "xlsx", "xls"])
    use_demo = st.checkbox("Utiliser le jeu de démonstration", value=up is None)
    st.divider()
    logo_up = st.file_uploader("Logo (optionnel)", type=["png", "jpg", "jpeg"])

if up is not None and not use_demo:
    df, bad_dates = load_file(up)
else:
    df, bad_dates = generate_demo_data(), {}

orch = Orchestrator(api_key, model, temperature)
issues = validate(df, bad_dates)
score = quality_score(len(df), issues)
n_decl = df["declarant"].nunique() if "declarant" in df else 0
has_dates = all(c in df.columns for c in ("declarant", "date_attendue", "date_transmission"))
batches = batch_table(df) if has_dates else pd.DataFrame()
on_time = float(batches["a_temps"].mean() * 100) if len(batches) else float("nan")
crit = int((issues["severite"] == "Critique").sum())

# ---- En-tête
if logo_up:
    st.image(Image.open(logo_up), width=170)
st.markdown("""
<div class="hero">
  <div class="brand">CREDIT<b>INFO</b><small>CENTRAL AFRICA</small></div>
  <div><h1>BIC-Guard AI</h1>
  <p>Contrôle, validation et reporting des données de crédit transmises par les déclarants — zone CEMAC</p></div>
</div>""", unsafe_allow_html=True)

items = [f"{len(df):,} enregistrements analysés".replace(",", " "),
         f"{n_decl} déclarants", f"Score qualité {score:.0f}/100",
         f"{crit} anomalies critiques", f"{len(issues)} anomalies au total"]
if len(batches):
    items.append(f"{on_time:.0f} % des transmissions à temps")
    worst = batches.groupby("declarant")["retard_j"].mean().idxmax()
    items.append(f"Plus gros retard moyen : {worst}")
html = "".join(f"<span>{i}</span>" for i in items)
st.markdown(f'<div class="ticker"><div class="ticker-track">{html}{html}</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{score:.0f}", "Score de qualité des données (/100)")
kpi(c2, f"{crit}", "Anomalies critiques à traiter", dark=True)
kpi(c3, "—" if np.isnan(on_time) else f"{on_time:.0f} %", "Transmissions dans les délais")
kpi(c4, f"{n_decl}", "Déclarants suivis", dark=True)
st.write("")

tab0, tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Agent 1 · Validateur & Auditeur",
                                  "Agent 2 · Assistant déclarants", "Agent 3 · Reporting exécutif"])

# ---- Vue d'ensemble
with tab0:
    st.subheader("Aperçu des données transmises")
    st.dataframe(df.head(200), width="stretch", height=300)
    if len(issues):
        a, b = st.columns(2)
        by_sev = issues.groupby("severite").size().reindex(["Critique", "Majeure", "Mineure"]).fillna(0).reset_index(name="n")
        a.plotly_chart(style_fig(px.bar(by_sev, x="severite", y="n", title="Anomalies par sévérité",
                                        color="severite", color_discrete_map={"Critique": DARK_RED, "Majeure": RED, "Mineure": "#E8707F"})),
                       width="stretch")
        by_rule = issues.groupby("regle").size().sort_values().reset_index(name="n").tail(10)
        b.plotly_chart(style_fig(px.bar(by_rule, x="n", y="regle", orientation="h", title="Règles les plus déclenchées")),
                       width="stretch")

# ---- Agent 1
with tab1:
    st.markdown('<div class="agent-box"><b>Agent 1 — Data Quality.</b> Contrôles de cohérence, omissions, doublons, '
                'valeurs atypiques ; matrice de risques par déclarant puis rapport d\'audit généré par le LLM.</div>',
                unsafe_allow_html=True)
    st.write("")
    if issues.empty:
        st.success("Aucune anomalie détectée sur ce fichier.")
    else:
        matrix = issues.pivot_table(index="declarant", columns="categorie", values="regle", aggfunc="count", fill_value=0)
        st.plotly_chart(style_fig(px.imshow(matrix, color_continuous_scale=RED_SCALE, text_auto=True, aspect="auto",
                                            title="Matrice de risques — déclarant × catégorie d'anomalie"), 380),
                        width="stretch")
        f1, f2 = st.columns(2)
        sev_f = f1.multiselect("Sévérité", ["Critique", "Majeure", "Mineure"], default=["Critique", "Majeure"])
        decl_f = f2.multiselect("Déclarant", sorted(issues["declarant"].unique()))
        view = issues[issues["severite"].isin(sev_f)]
        if decl_f:
            view = view[view["declarant"].isin(decl_f)]
        st.dataframe(view, width="stretch", height=300)
        st.download_button("Télécharger le journal d'anomalies (CSV)", issues.to_csv(index=False, sep=";").encode("utf-8-sig"),
                           "journal_anomalies.csv", "text/csv")

    summary = {"lignes": len(df), "score_qualite": score,
               "anomalies_par_severite": issues["severite"].value_counts().to_dict(),
               "anomalies_par_regle": issues["regle"].value_counts().head(10).to_dict(),
               "anomalies_par_declarant": issues["declarant"].value_counts().head(8).to_dict(),
               "exemples": issues.groupby("regle").head(2).head(12)[["declarant", "id_contrat", "regle"]].to_dict("records")}
    st.session_state["audit_context"] = summary
    if st.button("Générer le rapport d'audit (Agent 1)", disabled=issues.empty):
        if not orch.ready:
            st.warning("Renseignez une clé API Groq dans la barre latérale pour activer l'analyse LLM.")
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    st.session_state["audit_text"] = orch.run("validateur", summary=summary)
                except Exception as e:
                    llm_error(e)
    if st.session_state.get("audit_text"):
        st.markdown(st.session_state["audit_text"])

# ---- Agent 2
with tab2:
    st.markdown('<div class="agent-box"><b>Agent 2 — Support déclarants.</b> Posez vos questions sur la génération des '
                'fichiers, les règles de transmission ou une erreur d\'intégration. L\'agent connaît le référentiel '
                'ci-dessous et le dernier contrôle effectué.</div>', unsafe_allow_html=True)
    with st.expander("Référentiel de transmission (modifiable)"):
        referentiel = st.text_area("Référentiel", DEFAULT_REFERENTIEL, height=240, label_visibility="collapsed")
    hist = st.session_state.setdefault("chat", [])
    q1, q2, q3 = st.columns(3)
    preset = None
    if q1.button("Quel format de fichier envoyer ?"):
        preset = "Quel format de fichier dois-je envoyer et quels champs sont obligatoires ?"
    if q2.button("Expliquer mes erreurs"):
        preset = "Explique-moi les principales erreurs du dernier contrôle et comment les corriger dans mon extraction."
    if q3.button("Délais de transmission"):
        preset = "Quelles sont les échéances de transmission et que se passe-t-il en cas de retard ?"
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt = st.chat_input("Votre question à l'assistant…") or preset
    if prompt:
        hist.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if not orch.ready:
                reply = "Clé API Groq absente : renseignez-la dans la barre latérale pour activer l'assistant."
                st.warning(reply)
            else:
                try:
                    reply = st.write_stream(orch.run("assistant", history=hist[-10:], referentiel=referentiel,
                                                     context=st.session_state.get("audit_context", {})))
                except Exception as e:
                    reply = "Erreur lors de l'appel au modèle."
                    llm_error(e)
        hist.append({"role": "assistant", "content": reply})
    if hist and st.button("Effacer la conversation"):
        st.session_state["chat"] = []
        st.rerun()

# ---- Agent 3
with tab3:
    st.markdown('<div class="agent-box"><b>Agent 3 — Reporting exécutif.</b> Ponctualité, retards récurrents, qualité '
                'par déclarant et note de décision pour la direction.</div>', unsafe_allow_html=True)
    st.write("")
    if batches.empty:
        st.info("Colonnes requises pour ce rapport : declarant, date_attendue, date_transmission.")
    else:
        per_decl = batches.groupby("declarant").agg(retard_moyen=("retard_j", "mean"),
                                                    taux_a_temps=("a_temps", "mean"),
                                                    transmissions=("periode", "count")).reset_index()
        anom = issues.groupby("declarant").size().rename("anomalies")
        vol = df.groupby("declarant").size().rename("lignes")
        per_decl = per_decl.merge(anom, on="declarant", how="left").merge(vol, on="declarant").fillna({"anomalies": 0})
        per_decl["taux_anomalies_%"] = (per_decl["anomalies"] / per_decl["lignes"] * 100).round(1)
        per_decl["taux_a_temps"] = (per_decl["taux_a_temps"] * 100).round(0)
        per_decl["retard_moyen"] = per_decl["retard_moyen"].round(1)
        recurrents = batches[batches["retard_j"] > 0].groupby("declarant").size()
        per_decl["retards_recurrents"] = per_decl["declarant"].map(recurrents).fillna(0).astype(int)

        g1, g2 = st.columns([1, 2])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=on_time, number={"suffix": " %"},
                                       title={"text": "Conformité délais (global)"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED},
                                              "steps": [{"range": [0, 60], "color": "#F4B6BF"},
                                                        {"range": [60, 85], "color": "#F8E1E4"},
                                                        {"range": [85, 100], "color": "#FFFFFF"}]}))
        g1.plotly_chart(style_fig(gauge, 340), width="stretch")
        ps = per_decl.sort_values("retard_moyen")
        g2.plotly_chart(style_fig(px.bar(ps, x="retard_moyen", y="declarant", orientation="h",
                                         color="retard_moyen", color_continuous_scale=RED_SCALE,
                                         title="Retard moyen de transmission (jours, négatif = en avance)"), 340),
                        width="stretch")
        heat = batches.pivot(index="declarant", columns="periode", values="retard_j")
        st.plotly_chart(style_fig(px.imshow(heat, color_continuous_scale=RED_SCALE, aspect="auto", text_auto=True,
                                            title="Historique des retards — déclarant × période (jours)"), 380),
                        width="stretch")
        trend = batches.groupby("periode")["a_temps"].mean().mul(100).reset_index()
        st.plotly_chart(style_fig(px.line(trend, x="periode", y="a_temps", markers=True,
                                          title="Évolution du taux de transmission à temps (%)"), 300),
                        width="stretch")
        st.dataframe(per_decl.sort_values("retard_moyen", ascending=False), width="stretch")

        metrics = {"score_qualite": score, "taux_a_temps_global_%": round(on_time, 1),
                   "anomalies_critiques": crit, "declarants": per_decl.to_dict("records"),
                   "tendance_a_temps_%": dict(zip(trend["periode"], trend["a_temps"].round(0)))}
        if st.button("Générer la note de décision (Agent 3)"):
            if not orch.ready:
                st.warning("Renseignez une clé API Groq dans la barre latérale pour activer la synthèse LLM.")
            else:
                with st.spinner("Rédaction de la synthèse…"):
                    try:
                        st.session_state["report_text"] = orch.run("reporting", metrics=metrics)
                    except Exception as e:
                        llm_error(e)
        if st.session_state.get("report_text"):
            st.markdown(st.session_state["report_text"])
            st.download_button("Télécharger la note (Markdown)", st.session_state["report_text"].encode("utf-8"),
                               "note_decision_bic.md", "text/markdown")
