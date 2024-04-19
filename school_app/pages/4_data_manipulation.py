import pandas as pd
import streamlit as st


def afficher_data_manipulation():
    # Personnalisation de l'affichag
    st.title("Data Manipulation")

    # Charger le fichier CSV
    df = pd.read_csv("school_app/data.csv")


    # Excel étant une matière facultative alors nous pouvons remplacer les valeurs manquantes par 0
    df['excel'] = df['excel'].fillna(0)
    st.success("1. Les notes manquantes dans la colonne Excel ont été remplacé par 0")


    # Convertion de la colonne 'date' en type datetime
    df['date'] = pd.to_datetime(df['date'])
    st.success("2. Les dates dans la colonne date ont été converties de type object à datetime")


    # Suppression des valeurs dupliquées
    df = df.drop_duplicates()
    st.error("**3. Suppression des valeurs dupliquées**")


    # Gestion des valeurs manquantes
    for col in df.columns:
        if df[col].dtype == 'object':
            # Si la colonne est catégorielle, remplacez les valeurs manquantes par le mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

        elif df[col].dtype in ['int64', 'float64']:
            # Sinon, si c'est numérique, remplacez les valeurs manquantes par 0
            mean_val = 0
            df[col].fillna(mean_val, inplace=True)

        else:
            pass

    st.info("""4. Les valeurs manquantes dans les colonnes catégorielles sont remplacées par
              le mode de chaque colonne. Les valeurs manquantes dans les colonnes numériques 
              sont remplacées par 0, car ce sont des notes d'évaluations.""")


    # Création de nouvelles caractéristiques en fonction de l'analyse exploratoire.

    ## Création de la colonne 'moyenne' basée sur la notes des différentes compétences
    df['moyenne'] = (df['sql']*2 + df['excel']*0.05 + df['maths']*4 + df['python']*5 + \
                    df['poo']*2 + df['tableau']*2 + df['data_exploration']*2 + \
                    df['data_manipulation']*3 + df['data_viz']*2 + df['data_transformation'] + \
                    df['data_modelisation']*6 + df['data_deployement']*5 + \
                      df['pack_office'] + df['result_presentation']*5) / 40

    st.success("5. La colonne **moyene** a été ajouté au dataframe et comporte la moyenne obtenue "
             "à partir des notes des notes d'evaluation")


    # Ajout de la colonne "admis" à df
    df['admis'] = df['moyenne'].apply(lambda x: 'oui' if x >= 50 else 'non')
    st.info("6. La colonne **admis** a été ajouté au dataframe pour identifier les admis selon la moyenne obtenue")


    # Ajout de la colonne mention au dataframe
    df['mention'] = df['moyenne'].apply(lambda x: 'refusée' if x < 50 else
                                              ('passable' if x < 60 else
                                               ('assez bien' if x < 70 else
                                                ('bien' if x < 80 else
                                                 ('très bien' if x < 90 else 'excellent')))))
    st.success("7. La colonne **mention** est ajoutée avec succès au dataframe selon la valeur de la moyenne")


    # Enregistrement de l'année dans une colonne séparée
    df['annee'] = df['date'].dt.year
    st.info("8. L'année est enregistrée dans une colonne séparée pour faciliter la visualisation des données par année")


    # Enregistrer les données prétraitées
    df.to_pickle("data_preprocessed.pkl")
    st.info("9. Les données prétraitées sont enregistrées dans **data_preprocessed** au format pickle afin de l'utiliser par la suite")

    # Naviguer entre les sections
    st.header("Accéder à une autre section dans un nouvel onglet")

    # Liste des liens vers les autres sections
    st.write("""
    - [Information](http://localhost:8501/Information)
    - [Exploration des données](http://localhost:8501/Data_Exploration)
    - [Manipulation des données](http://localhost:8501/Data_Manipulation)
    - [Prise de décision guidée](http://localhost:8501/Driven_Decision)
    - [Visualisation des données](http://localhost:8501/Data_Visualisation)
    - [Modélisation des données](http://localhost:8501/Data_Modelisation)
    - [Nous contacter](http://localhost:8501/Nous_contacter)
    - [À propos de nous](http://localhost:8501/About_us)
    """)
afficher_data_manipulation()



