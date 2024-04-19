import pandas as pd
import streamlit as st


def afficher_data_exploration():
    # Personnalisation de l'affichag
    st.title("Data Exploration")

    # Charger le fichier CSV
    df = pd.read_csv("school_app/data.csv")

    # Information sur les données

    ## Le nombre de ligne et nombre de colonne
    st.write("### Forme du dataframe : ")
    line, col = df.shape
    st.write(f""" \nLe nombre de ligne dans le dataframe est : {line}
                  \nLe nombre de colonne dans le dataframe est: {col}
    """)


    # Vérificattion des valeurs manquantes
    valeurs_manquantes = df.isnull().sum()
    valeurs_manquantes = pd.DataFrame(valeurs_manquantes).T
    st.write("### **Valeurs manquantes par colonne :** \n", valeurs_manquantes)


    # Vérification des doublons
    doublons = df[df.duplicated()]
    st.write("### Données dupliquées :", doublons)


    # Vérification des valeurs aberrantes
    st.write("### **Les valeurs aberantes :**")
    # Définir le seuil pour identifier les valeurs aberrantes en fonction du score Z
    seuil_z = 3

    # Boucle à travers chaque colonne
    for colonne in df.select_dtypes(include=['int64', 'float64']).columns:
        # Calculer la moyenne et l'écart-type de la colonne actuelle
        moyenne = df[colonne].mean()
        ecart_type = df[colonne].std()

        # Calculer les scores Z pour la colonne actuelle
        scores_z = (df[colonne] - moyenne) / ecart_type

        # Identifier les valeurs aberrantes pour la colonne actuelle
        valeurs_aberrantes = df[abs(scores_z) > seuil_z]

        # Afficher les valeurs aberrantes pour la colonne actuelle
        if not valeurs_aberrantes.empty:
            st.write(f"Valeurs aberrantes dans la colonne '{colonne}':")
            for index, valeur in valeurs_aberrantes[colonne].items():
                st.write(f"Ligne: {index}, Valeur: {valeur}")


    # Statistiques descriptives pour résumer les caractéristiques de chaque variable.
    st.write("### **Statistiques descriptives (sur les variables numériques)**")
    st.write(df.describe())


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
afficher_data_exploration()
