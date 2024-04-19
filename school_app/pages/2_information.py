import pandas as pd
import streamlit as st

def afficher_information():
    # Personnalisation de l'affichage
    st.title("Informations sur ces données")
    st.warning("CE SEONT DES DONNEES FICTIVES GENEREES, PAS DE DONNEES REELLES")

    # Charger le fichier CSV
    df = pd.read_csv("school_app/data.csv")

    # Affichage du dataframe
    st.write(df)


    # Afficher le nom des colonnes
    st.subheader("Information sur les noms des colonnes")
    st.write("""
    **student_id** : contient l'identifiant unique de chaque étudiant et doit commencer par "GMC0000001" puis ""GMC0000002" jusqu'à ""GMC0002000"
    
    **date** : contient la date d'obtention du diplôme (les 5 derniers jours ouvrables du mois de juin 2020, 2021, 2022, 2023). crée d'abord 367 lignes pour 2020, ensuite 433 lignes pour 2021, puis 549 pour l'année 2022 et 651 ligne pour l'année 2023.
    
    **name**: contient aléatoirement de vrai noms Ivoiriens, Béninois, Camerounais, Tunisiens et Sénégalais
    
    **country** : Contient le pays d'origine des étudiants (Côte d'ivoire pour les Ivoiriens, Bénin pour les Béninois, Cameroun pour les camerounais, Tunisie pour les tunisiens et Sénégal pour les Sénégalais)
    
    **Les autres colonnes** : sql, excel, maths, python, poo, tableau, data_exploration, data_manipulation, data_viz, data_transformation, data_modelisation, data_deployement, pack_office, result_presentation contiennent des notes obtenues pour l'évaluation.
    """)


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

afficher_information()
