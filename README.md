# BIC-Guard AI

Plateforme agentique de contrôle, validation et reporting pour Bureau d'Information sur le Crédit
(Creditinfo Central Africa — zone CEMAC). Auteur : Romuald Arthur MBANA MEDJO.

3 agents IA (Llama via Groq) pilotés par un orchestrateur : Validateur & Auditeur, Assistant déclarants,
Reporting exécutif.

## Prérequis
- Python 3.10 ou supérieur
- Une clé API Groq (gratuite sur https://console.groq.com)

## Installation et lancement rapides
- Windows : double-clic sur `run.bat`
- Linux / macOS : `chmod +x run.sh && ./run.sh`

Les scripts créent un environnement virtuel `.venv`, installent `requirements.txt` puis lancent l'application.

## Installation manuelle
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Clé API Groq
Soit via la barre latérale de l'application, soit :
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis renseigner GROQ_API_KEY
```

## Modèle
Par défaut : `llama-3.3-70b-versatile`. Le champ « Modèle » de la barre latérale accepte tout modèle
disponible sur votre compte Groq (y compris un Llama 3.2). Voir https://console.groq.com/docs/models

## Données
- Sans fichier : jeu de démonstration (8 déclarants, 6 mois, anomalies injectées).
- Avec fichier CSV/XLSX : colonnes attendues
  `declarant, id_contrat, id_client, type_credit, montant, encours, taux, date_octroi, date_echeance, statut, date_attendue, date_transmission`.
  Un exemple est fourni dans `exemple_declarant.csv`.

## Structure
```
app.py                       Application (orchestrateur + 3 agents + UI)
requirements.txt             Dépendances
run.sh / run.bat             Installation + lancement
exemple_declarant.csv        Fichier d'exemple
.streamlit/config.toml       Thème Creditinfo
.streamlit/secrets.toml.example
```
