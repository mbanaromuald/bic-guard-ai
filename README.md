# BIC-Guard AI

**Plateforme agentique de contrôle, validation et reporting pour Bureau d'Information sur le Crédit (BIC)**

- **Auteur** : Romuald Arthur MBANA MEDJO
- **Contexte visé** : Creditinfo Central Africa (Douala, Cameroun / zone CEMAC) — poste de Junior Data Manager
- **Stack** : Python 3.10+, Streamlit, Pandas, Plotly, Groq (inférence) + modèles Llama
- **Statut** : prototype fonctionnel (démonstrateur). Voir la section [Limites](#9-limites-actuelles) avant tout usage en production.

---

## Sommaire

1. [Pourquoi ce projet](#1-pourquoi-ce-projet)
2. [Vue d'ensemble](#2-vue-densemble)
3. [Architecture](#3-architecture)
4. [Les trois agents en détail](#4-les-trois-agents-en-détail)
5. [Données : format attendu et jeu de démonstration](#5-données--format-attendu-et-jeu-de-démonstration)
6. [Installation et lancement](#6-installation-et-lancement)
7. [Guide d'utilisation](#7-guide-dutilisation)
8. [Avantages](#8-avantages)
9. [Limites actuelles](#9-limites-actuelles)
10. [Perspectives et feuille de route](#10-perspectives-et-feuille-de-route)
11. [Sécurité et confidentialité](#11-sécurité-et-confidentialité)
12. [Déploiement (GitHub / Streamlit Cloud)](#12-déploiement-github--streamlit-cloud)
13. [Dépannage](#13-dépannage)
14. [Structure du dépôt](#14-structure-du-dépôt)
15. [Charte graphique](#15-charte-graphique)

---

## 1. Pourquoi ce projet

Un Bureau d'Information sur le Crédit collecte chaque mois, auprès de banques et d'institutions de microfinance (les *déclarants*), des fichiers décrivant les contrats de crédit de leurs clients. La valeur d'un BIC repose entièrement sur **la qualité, la complétude et la ponctualité** de ces données : une erreur de saisie, un doublon ou une transmission en retard dégradent directement la fiabilité des rapports de crédit consultés ensuite par les prêteurs.

Dans la pratique, le Data Manager d'un BIC fait face à trois problèmes récurrents :

| Problème | Conséquence |
|---|---|
| Contrôler à la main (ou par scripts ad hoc) des fichiers volumineux et hétérogènes | Temps perdu, anomalies manquées, critères non homogènes d'un déclarant à l'autre |
| Répondre en continu aux questions techniques des déclarants (format, champs, erreurs d'intégration) | Charge de support élevée, réponses variables selon l'interlocuteur |
| Produire des synthèses de conformité pour la direction | Reporting chronophage, retards chroniques identifiés tardivement |

**BIC-Guard AI** adresse ces trois points avec trois agents spécialisés. Le choix de conception central est le suivant : **les calculs et les contrôles sont déterministes (Pandas), le modèle de langage sert à interpréter, expliquer et rédiger**. Les chiffres ne sont donc jamais « inventés » par l'IA : ils viennent du code, et le LLM les met en forme et les commente.

---

## 2. Vue d'ensemble

L'application est une interface web unique (Streamlit) organisée en quatre onglets :

| Onglet | Contenu |
|---|---|
| **Vue d'ensemble** | Aperçu des données, anomalies par sévérité, règles les plus déclenchées |
| **Agent 1 · Validateur & Auditeur** | Contrôles de qualité, matrice de risques, journal d'anomalies exportable, rapport d'audit rédigé par le LLM |
| **Agent 2 · Assistant déclarants** | Chat de support basé sur un référentiel de transmission modifiable et sur le dernier contrôle effectué |
| **Agent 3 · Reporting exécutif** | Ponctualité, retards récurrents, qualité par déclarant, note de décision pour la direction |

Au-dessus des onglets, un **bandeau défilant** et quatre **KPI** (score de qualité, anomalies critiques, transmissions dans les délais, nombre de déclarants) donnent l'état global en un coup d'œil.

---

## 3. Architecture

```
                    ┌───────────────────────────┐
                    │  Données déclarants       │
                    │  (CSV / XLSX / démo)      │
                    └─────────────┬─────────────┘
                                  │  chargement, typage, parsing des dates
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │   Couche déterministe (Pandas)                     │
        │   validate()  ·  quality_score()  ·  batch_table() │
        └─────────────┬──────────────────────────────────────┘
                      │  résultats chiffrés (JSON de synthèse)
                      ▼
        ┌────────────────────────────────────────────────────┐
        │              ORCHESTRATEUR CENTRAL                 │
        │  classe Orchestrator : run(agent, **kw)            │
        │  • un seul client Groq partagé                     │
        │  • modèle, température, limites de tokens          │
        └──────┬─────────────────┬─────────────────┬─────────┘
               ▼                 ▼                 ▼
        ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
        │  AGENT 1    │   │  AGENT 2    │   │  AGENT 3    │
        │ Validateur  │   │ Assistant   │   │ Reporting   │
        │ & Auditeur  │   │ déclarants  │   │ & Analytics │
        └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
               └─────────────────┼─────────────────┘
                                 ▼
                    ┌───────────────────────────┐
                    │  Groq  →  modèle Llama    │
                    │  (API chat completions)   │
                    └───────────────────────────┘
                                 │
                                 ▼
               Interface Streamlit : tableaux, graphiques Plotly,
               exports CSV / Markdown, chat en streaming
```

### Principes de conception

1. **Séparation calcul / langage.** Les règles, scores, ratios et délais sont calculés en Python. Le LLM reçoit uniquement des résultats agrégés et a pour consigne explicite de ne rien inventer.
2. **Dégradation gracieuse.** Sans clé API, l'application reste utilisable : contrôles, matrice de risques, graphiques et exports fonctionnent. Seules les parties rédigées par le LLM sont désactivées, avec un message clair.
3. **Orchestration centralisée.** Toute la logique d'accès au modèle (clé, modèle, température, gestion du flux) est concentrée dans la classe `Orchestrator`. Changer de modèle ou de fournisseur ne touche qu'un endroit.
4. **Contexte partagé.** Le résumé du dernier contrôle de l'Agent 1 est transmis à l'Agent 2, qui peut ainsi expliquer des erreurs concrètes plutôt que des généralités.

> **Précision sur le terme « agentique ».** Dans cette version, l'orchestrateur est un *répartiteur* : l'utilisateur choisit l'agent (onglet), l'orchestrateur exécute la tâche correspondante. Il ne planifie pas de manière autonome et n'enchaîne pas des outils de lui-même. Voir [Perspectives](#10-perspectives-et-feuille-de-route) pour l'évolution vers une orchestration autonome.

---

## 4. Les trois agents en détail

### 4.1 Agent 1 — Validateur & Auditeur de données (Data Quality)

**Rôle** : contrôler la cohérence d'un fichier transmis, détecter anomalies et omissions, et produire un avis d'audit.

**Étape 1 — Contrôles déterministes** (fonction `validate`). Chaque règle produit des lignes de journal avec : numéro de ligne (correspondant à la ligne du fichier tableur, en-tête = ligne 1), déclarant, identifiant de contrat, règle, catégorie, sévérité, détail.

| Règle | Catégorie | Sévérité | Logique |
|---|---|---|---|
| Colonne obligatoire absente | Structure | Critique | Compare aux 12 colonnes requises |
| Valeur manquante (`id_contrat`, `id_client`, `montant`, `type_credit`, `statut`) | Omission | Majeure | Cellule vide |
| Doublon d'`id_contrat` | Doublon | Critique | Même couple (déclarant, id_contrat) déjà vu |
| Montant ≤ 0 | Cohérence | Critique | Capital négatif ou nul |
| Encours > montant | Cohérence | Majeure | Encours supérieur au capital octroyé |
| Encours négatif | Cohérence | Critique | — |
| Échéance ≤ octroi | Cohérence | Critique | Chronologie impossible |
| Taux hors plage | Atypique | Majeure | Taux < 0 % ou > 40 % (seuil paramétrable, constante `TAUX_MAX`) |
| Statut invalide | Format | Majeure | Hors {Normal, Impayé, Restructuré, Contentieux, Soldé} |
| Date illisible | Format | Majeure | Valeur non convertible en date (hors cellule vide) |
| Montant atypique | Atypique | Mineure | Montant > Q3 + 3 × IQR au sein de son type de crédit |

**Étape 2 — Score de qualité (0 à 100).**
Chaque anomalie pèse 3 (critique), 2 (majeure) ou 1 (mineure). La pénalité est le total pondéré rapporté au nombre de lignes (× 3), mis à l'échelle ×4 :

```
score = max(0, 100 − (Σ poids / (nb_lignes × 3)) × 100 × 4)
```

C'est un indicateur de pilotage relatif (comparer deux fichiers, suivre une évolution), pas une norme réglementaire. Sur le jeu de démonstration, il est d'environ 75/100.

**Étape 3 — Matrice de risques.** Heatmap déclarant × catégorie d'anomalie : on voit immédiatement quel déclarant pose quel type de problème.

**Étape 4 — Rapport d'audit par le LLM.** Le modèle reçoit un JSON de synthèse (volumes, répartition par sévérité, par règle, par déclarant, quelques exemples) et rédige : synthèse en 3 lignes, risques prioritaires, actions correctives par déclarant, décision recommandée (*Accepter* / *Accepter sous réserve* / *Retourner au déclarant*) avec justification.

**Exports** : journal d'anomalies en CSV (séparateur `;`, encodage UTF-8 avec BOM pour une ouverture directe dans Excel).

### 4.2 Agent 2 — Assistant Support Déclarants (Declarant Onboarding)

**Rôle** : accompagner banques et IMF sur la génération et la transmission des fichiers.

- **Chat en streaming**, historique conservé pendant la session (les 10 derniers messages sont renvoyés au modèle).
- **Référentiel de transmission modifiable** dans l'interface (format de fichier, champs obligatoires, formats de dates, règles d'intégrité, échéances, statuts admis). Le modèle a pour consigne de s'y tenir et, si l'information n'y figure pas, de le dire et de renvoyer vers l'équipe Creditinfo plutôt que d'improviser.
- **Contexte du dernier contrôle** : l'agent connaît le résumé de l'audit de l'Agent 1, ce qui lui permet d'expliquer « pourquoi mon fichier a été rejeté » et « comment corriger mon extraction ».
- **Questions rapides** : trois boutons (format de fichier, explication des erreurs, délais de transmission).

> Le référentiel fourni par défaut est **illustratif**. Il doit être remplacé par les spécifications officielles de Creditinfo avant tout usage réel.

### 4.3 Agent 3 — Reporting & Analytics Exécutif

**Rôle** : mesurer la conformité des déclarants dans le temps et formuler des synthèses de décision.

**Calculs** (fonction `batch_table` et agrégations) :
- Un *lot* = un déclarant pour une période (mois de référence). Le **retard** est la médiane, sur le lot, de `date_transmission − date_attendue` en jours (négatif = en avance). Un lot est **à temps** si son retard ≤ 0.
- Par déclarant : retard moyen, taux de transmissions à temps, nombre de transmissions, nombre de **retards récurrents** (lots en retard), nombre et **taux d'anomalies** (anomalies / lignes transmises).

**Visualisations Plotly** : jauge de conformité globale, retard moyen par déclarant, heatmap déclarant × période, évolution mensuelle du taux de ponctualité, tableau récapitulatif.

**Note de décision par le LLM** : constats clés, déclarants à risque, retards récurrents, tendance, recommandations (relances, accompagnement, escalade), 250 mots maximum. Exportable en Markdown.

---

## 5. Données : format attendu et jeu de démonstration

### 5.1 Colonnes attendues (une ligne par contrat)

| Colonne | Type | Description |
|---|---|---|
| `declarant` | texte | Nom de la banque / IMF |
| `id_contrat` | texte | Identifiant unique du contrat (unique par déclarant) |
| `id_client` | texte | Identifiant du client |
| `type_credit` | texte | Consommation, Immobilier, Équipement, Trésorerie, Microcrédit… |
| `montant` | nombre | Capital octroyé (XAF, sans séparateur de milliers) |
| `encours` | nombre | Capital restant dû (XAF) |
| `taux` | nombre | Taux annuel en pourcentage |
| `date_octroi` | date | AAAA-MM-JJ |
| `date_echeance` | date | AAAA-MM-JJ |
| `statut` | texte | Normal, Impayé, Restructuré, Contentieux, Soldé |
| `date_attendue` | date | Date limite de transmission du lot |
| `date_transmission` | date | Date effective de transmission |

Formats acceptés : **CSV** (séparateur détecté automatiquement) et **XLSX/XLS**. Les noms de colonnes sont normalisés en minuscules. Un exemple est fourni : `exemple_declarant.csv`.

### 5.2 Jeu de démonstration

Généré de façon **synthétique** (aucune donnée réelle) : 8 déclarants fictifs, 6 périodes (avril à septembre 2026), environ 600 contrats. Chaque déclarant a un profil de ponctualité différent (de l'avance régulière au retard chronique) et des **anomalies volontairement injectées** : doublons, montants négatifs, encours supérieurs au montant, échéances antérieures à l'octroi, taux aberrants, statuts inconnus, identifiants clients manquants, montants atypiques. Cela permet de tester l'ensemble des fonctions sans aucun fichier.

---

## 6. Installation et lancement

### Prérequis
- Python **3.10 ou supérieur**
- Une clé API Groq (gratuite sur <https://console.groq.com>) — optionnelle pour tester les contrôles et graphiques

### Démarrage rapide
- **Windows** : double-clic sur `run.bat`
- **Linux / macOS** : `chmod +x run.sh && ./run.sh`

Les scripts créent un environnement virtuel `.venv`, installent `requirements.txt` et lancent l'application.

### Installation manuelle
```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Dépendances (`requirements.txt`)
`streamlit>=1.50.0`, `groq>=0.9.0`, `pandas>=2.0.0`, `plotly>=5.18.0`, `pillow>=10.0.0`, `openpyxl>=3.1.0` (lecture des fichiers Excel).

### Clé API Groq
Deux options :
1. Saisie dans la barre latérale de l'application ;
2. Fichier `.streamlit/secrets.toml` (copier `secrets.toml.example`) :
   ```toml
   GROQ_API_KEY = "votre_cle_api_groq"
   ```

### Choix du modèle
Le modèle par défaut est `llama-3.3-70b-versatile`. Le champ **Modèle** de la barre latérale accepte n'importe quel identifiant disponible sur votre compte Groq (par exemple `llama-3.1-8b-instant`, ou un modèle Llama 3.2 si votre compte y a accès). Le catalogue Groq évolue : consultez <https://console.groq.com/docs/models>. Un modèle plus petit est plus rapide et moins coûteux en quota, un modèle plus grand rédige généralement des synthèses plus nuancées.

---

## 7. Guide d'utilisation

1. **Choisir la source de données** (barre latérale) : charger un fichier CSV/XLSX, ou laisser coché « Utiliser le jeu de démonstration ».
2. **Lire le bandeau et les KPI** : score de qualité, anomalies critiques, ponctualité, nombre de déclarants.
3. **Onglet Vue d'ensemble** : aperçu des 200 premières lignes, répartition des anomalies.
4. **Onglet Agent 1** :
   - consulter la matrice de risques ;
   - filtrer le journal par sévérité et par déclarant, puis l'exporter ;
   - cliquer sur **Générer le rapport d'audit** pour obtenir l'avis rédigé.
5. **Onglet Agent 2** : adapter si besoin le référentiel, puis poser une question ou utiliser un bouton rapide. Lancer d'abord l'Agent 1 permet à l'assistant d'expliquer des erreurs réelles.
6. **Onglet Agent 3** : examiner les graphiques de ponctualité, puis **Générer la note de décision** et la télécharger.
7. **Réglages utiles** : la **température** (0,2 par défaut) contrôle la variabilité des textes ; une valeur basse favorise des réponses stables et factuelles. Un **logo** peut être chargé pour personnaliser l'en-tête.

---

## 8. Avantages

**Qualité et fiabilité**
- **Contrôles reproductibles** : les mêmes règles s'appliquent à tous les déclarants, sans variation d'un opérateur à l'autre.
- **Chiffres non hallucinés** : tous les indicateurs sont calculés par le code ; le LLM reçoit des résultats et a pour consigne de ne rien inventer.
- **Traçabilité** : chaque anomalie est localisée (ligne, déclarant, contrat, règle, sévérité) et exportable.

**Gain de temps**
- Un audit complet d'un fichier se fait en quelques secondes, rapport rédigé inclus.
- L'Agent 2 absorbe une partie des questions récurrentes de support et les explique à partir des erreurs réelles du fichier.
- La note de décision pour la direction est produite en un clic.

**Pilotage**
- Vision synthétique par déclarant (matrice de risques, ponctualité, taux d'anomalies) pour prioriser relances et accompagnements.
- Score de qualité comparable dans le temps.

**Technique et coût**
- **Inférence rapide** via Groq : réponses quasi instantanées, chat fluide en streaming.
- **Coût d'entrée très faible** : clé Groq gratuite, aucun serveur dédié, une seule dépendance applicative (Streamlit).
- **Modèle interchangeable** : un seul champ pour changer de modèle Llama.
- **Prototype léger et lisible** : un fichier de code unique, facile à relire, adapter et étendre.
- **Dégradation gracieuse** : utilisable hors ligne côté contrôles (sans clé API).

**Adaptabilité**
- Règles, seuils, statuts et référentiel sont des constantes ou des champs modifiables, donc adaptables à la réglementation et aux spécifications réelles.

---

## 9. Limites actuelles

Cette section est volontairement franche : c'est un **prototype**, pas un système prêt pour la production.

### Limites fonctionnelles
- **Règles non validées réglementairement.** Les règles, le seuil de taux (40 %), le délai de transmission (le 10 du mois suivant) et le référentiel sont des **hypothèses illustratives**. Elles ne reflètent pas les spécifications officielles de Creditinfo ni la réglementation CEMAC/COBAC, qui doivent être intégrées par un expert métier.
- **Périmètre de contrôle limité.** Pas de rapprochement d'identités entre déclarants (un même client chez deux banques), pas de détection de doublons « flous » (identifiants légèrement différents), pas de contrôle de ratios d'endettement, pas de vérification de cohérence temporelle d'un mois sur l'autre pour un même contrat.
- **Unicité limitée au couple (déclarant, `id_contrat`)** au sein du fichier chargé.
- **Historique limité au fichier fourni.** Le suivi dans le temps (Agent 3) repose uniquement sur les périodes présentes dans les données chargées ; il n'y a pas de base d'historique persistante.
- **Score de qualité empirique.** La formule de pondération est un choix de conception, non calibré sur des données réelles.
- **Détection de valeurs atypiques simple** : méthode statistique univariée (IQR) par type de crédit, sans modèle multivarié.

### Limites liées au LLM
- **Risque d'erreur d'interprétation.** Bien que les chiffres viennent du code et que le prompt interdise d'inventer, un LLM peut mal interpréter, généraliser ou formuler de façon trop affirmative. Les textes générés sont une **aide à la décision à relire**, pas une décision.
- **Pas de recherche documentaire (RAG).** L'Agent 2 ne s'appuie que sur le référentiel collé dans l'interface et sur le résumé du dernier contrôle ; il ne consulte pas de base documentaire.
- **Orchestration non autonome.** Pas de planification ni d'appels d'outils décidés par le modèle (voir encadré section 3).
- **Aucune évaluation formelle.** La qualité des réponses des agents n'a pas fait l'objet d'un jeu de tests ou de métriques (précision, taux d'hallucination, satisfaction).
- **Dépendance à un service externe.** Disponibilité, quotas du palier gratuit, évolution du catalogue de modèles (certains modèles sont retirés) échappent au contrôle de l'application.
- **Langue** : interface et prompts en français uniquement.

### Limites techniques
- **Pas de persistance.** Tout est en mémoire de session : historique de chat, rapports, résultats sont perdus à la fermeture ou au rechargement de la page.
- **Pas d'authentification ni de rôles.** Toute personne accédant à l'application a accès à toutes les fonctions.
- **Passage à l'échelle non traité.** Les contrôles sont recalculés à chaque interaction et le journal est construit ligne à ligne ; c'est adapté à quelques milliers de lignes, pas à des fichiers de plusieurs centaines de milliers ou millions de lignes sans optimisation (vectorisation du journal, cache, traitement par lots).
- **Aucun test automatisé** (unitaires, intégration) ni intégration continue à ce stade.
- **Pas d'ingestion automatisée** : le chargement est manuel via l'interface (pas de dépôt SFTP, d'API ou de traitement planifié).
- **Affichage tronqué** : l'aperçu montre les 200 premières lignes.

### Limites de gouvernance
- **Confidentialité** : des données agrégées (et quelques identifiants de contrat dans les exemples) sont envoyées à un service tiers (Groq). Voir [section 11](#11-sécurité-et-confidentialité).
- **Pas de journal d'audit** des actions utilisateurs ni des textes générés.

---

## 10. Perspectives et feuille de route

### Court terme (consolidation)
- Intégrer les **règles officielles** Creditinfo/CEMAC : externaliser les règles dans un fichier de configuration (YAML/JSON) modifiable sans toucher au code.
- Ajouter des **tests unitaires** sur chaque règle et sur le score (pytest) et une **intégration continue** (GitHub Actions).
- **Optimiser les performances** : journal d'anomalies vectorisé, mise en cache (`st.cache_data`), traitement par lots.
- **Exports de rapports** en PDF/Word en plus du Markdown.
- Messages d'erreur et sélection de colonnes plus guidés au chargement d'un fichier au format différent (mappage de colonnes).

### Moyen terme (industrialisation)
- **Base de données persistante** (PostgreSQL) pour historiser transmissions, anomalies et rapports, et suivre l'évolution d'un déclarant sur de longues périodes.
- **Authentification et rôles** (Data Manager, direction, déclarant) avec un portail dédié aux déclarants.
- **Ingestion automatisée** : dépôt SFTP/API, traitement planifié, **alertes** (e-mail, messagerie) en cas de retard ou de seuil critique dépassé.
- **RAG** sur la documentation officielle (spécifications, circulaires, FAQ) pour que l'Agent 2 réponde avec citation de ses sources.
- **Vraie orchestration agentique** : appels d'outils (*function calling*), enchaînement autonome contrôle → diagnostic → relance → reporting, avec validation humaine aux étapes sensibles.
- **Détection d'anomalies par apprentissage automatique** (Isolation Forest, modèles multivariés) en complément des règles.
- **Rapprochement d'entités** (résolution d'identité des clients entre déclarants, doublons flous).
- **Évaluation systématique** des agents : jeu de cas de test, métriques de qualité, revue humaine.

### Long terme (valeur ajoutée métier)
- **Souveraineté des données** : exécution de modèles ouverts en local ou en cloud privé (par exemple via Ollama ou vLLM) pour qu'aucune donnée ne quitte l'infrastructure.
- **Analyses prédictives** : anticipation des retards de transmission et des dégradations de qualité par déclarant.
- **Scoring de crédit et indicateurs sectoriels** à partir des données consolidées du BIC.
- **Multilinguisme** (français/anglais) et **application mobile** de consultation pour la direction.
- **Journal d'audit complet** et conformité à la protection des données personnelles applicable.

---

## 11. Sécurité et confidentialité

- **Clé API** : ne jamais la commiter. Le fichier `.streamlit/secrets.toml` est exclu par `.gitignore`. Sur Streamlit Cloud, la clé se saisit dans *Settings → Secrets*.
- **Ce qui est envoyé à Groq** : des **synthèses agrégées** (volumes, répartitions, moyennes par déclarant) et un petit nombre d'exemples d'anomalies (déclarant, identifiant de contrat, règle). Les lignes complètes du fichier et les identifiants clients ne sont **pas** transmis. Le chat de l'Agent 2 envoie en outre vos messages, le référentiel et le résumé du dernier contrôle.
- **Données réelles** : avant d'utiliser des données de crédit réelles, il faut valider avec la conformité/le DPO l'envoi de données à un fournisseur tiers, ou passer à un modèle auto-hébergé (voir perspectives).
- **Application déployée en ligne** : sans authentification, toute personne disposant du lien y accède, et consomme le quota lié à la clé du déployeur. Pour une démonstration publique, utilisez le jeu de démonstration uniquement et envisagez de retirer la clé des secrets (chaque visiteur saisit alors la sienne).
- Les données chargées ne sont **pas enregistrées** sur disque par l'application (traitement en mémoire de session).

---

## 12. Déploiement (GitHub / Streamlit Cloud)

1. Placer le contenu du dossier à la **racine** d'un dépôt GitHub (`app.py` à la racine).
2. Pousser :
   ```bash
   git init && git add . && git commit -m "BIC-Guard AI"
   git branch -M main
   git remote add origin https://github.com/VOTRE_UTILISATEUR/bic-guard-ai.git
   git push -u origin main
   ```
3. Sur <https://share.streamlit.io> : **Create app** → choisir le dépôt, la branche `main`, le fichier `app.py`.
4. Dans **Advanced settings → Secrets**, ajouter `GROQ_API_KEY = "..."`, puis **Deploy**.
5. Chaque `git push` sur `main` redéploie automatiquement. Les applications gratuites se mettent en veille après inactivité (réveil en quelques secondes).

---

## 13. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| « Appel Groq impossible » | Clé invalide, modèle indisponible ou quota atteint | Vérifier la clé, changer le nom du modèle (catalogue Groq), patienter si limite de débit |
| Les boutons LLM affichent « Renseignez une clé API » | Clé absente | Saisir la clé dans la barre latérale ou dans `secrets.toml` |
| « Colonne manquante » en masse | Noms de colonnes différents de ceux attendus | Renommer les colonnes selon la [section 5.1](#51-colonnes-attendues-une-ligne-par-contrat) |
| Dates signalées illisibles | Format de date non standard | Utiliser AAAA-MM-JJ |
| Onglet Agent 3 vide avec un message d'info | Colonnes `declarant`, `date_attendue` ou `date_transmission` absentes | Ajouter ces colonnes |
| Erreur à la lecture d'un `.xlsx` | `openpyxl` absent | `pip install openpyxl` |
| Page lente avec un gros fichier | Contrôles recalculés à chaque interaction | Voir limites et perspectives (optimisation) |
| Erreur d'argument `width` sur les graphiques | Version de Streamlit trop ancienne | `pip install -U "streamlit>=1.50.0"` |

---

## 14. Structure du dépôt

```
bic-guard-ai/
├── app.py                          Application : données, règles, orchestrateur, 3 agents, interface
├── requirements.txt                Dépendances Python
├── run.sh / run.bat                Installation + lancement (Linux/macOS · Windows)
├── exemple_declarant.csv           Fichier d'exemple pour tester l'Agent 1
├── README.md                       Ce document
├── .gitignore                      Exclut .venv, __pycache__ et les secrets
└── .streamlit/
    ├── config.toml                 Thème Creditinfo
    └── secrets.toml.example        Modèle de configuration de la clé API
```

### Repères dans `app.py`
| Élément | Rôle |
|---|---|
| `generate_demo_data()` | Jeu de démonstration synthétique avec anomalies injectées |
| `load_file()` | Lecture CSV/XLSX, normalisation, typage, détection des dates illisibles |
| `validate()` / `quality_score()` | Règles de contrôle et score de qualité (Agent 1) |
| `batch_table()` | Calcul des lots et retards (Agent 3) |
| `Orchestrator` | Accès Groq, répartition `run()`, méthodes `audit`, `assist`, `report` |
| `SYS_AUDIT`, `SYS_REPORT` | Prompts système des Agents 1 et 3 (le prompt de l'Agent 2 est dans `assist`) |
| `DEFAULT_REFERENTIEL` | Référentiel de transmission illustratif de l'Agent 2 |

---

## 15. Charte graphique

| Élément | Valeur |
|---|---|
| Rouge Creditinfo | `#C8102E` |
| Rouge sombre / accent | `#8B0000` |
| Noir profond / texte | `#1A1A1A` |
| Gris clair / fond | `#F8F9FA` |

Éléments visuels : en-tête dégradé noir → rouge sombre → rouge, bandeau défilant de KPI (pause au survol, désactivé si l'utilisateur préfère réduire les animations), cartes KPI à fort contraste, graphiques Plotly interactifs aux teintes rouges. Le thème Streamlit est défini dans `.streamlit/config.toml`.

---

*BIC-Guard AI est un démonstrateur conçu pour illustrer l'apport d'agents IA à la gestion de la qualité des données d'un bureau d'information sur le crédit. Licence à définir par l'auteur.*
