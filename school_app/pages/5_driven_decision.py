import pickle as pk
import streamlit as st
import pandas as pd

def afficher_driven_decision():
    # Charger les données prétraitées dans data manpulation
    df = pk.load(open("data_preprocessed.pkl", "rb"))

    # Titre de la page
    st.title("Prise de décision en se basant sur les données")


    # Question 1
    st.write("**1. Quel est le pourcentage d'étudiants admis dans chaque pays ?**")

    # Pourcentage d'étudiants admis par pays
    admis_percentage = df[df['admis'] == 'oui'].groupby('country').size() / df.groupby('country').size() * 100
    st.write("Le pourcentage d'étudiants admis par pays :\n", pd.DataFrame(admis_percentage).T)


    # Question 2
    st.write("**2. Quelle est le nombre de mentions par type obtenues par les étudiants dans chaque pays ?**")
    # Regroupement des données par pays et mention
    pays_mention = df.groupby(['country', 'mention']).size().reset_index(name='count')

    # Création d'un tableau croisé pour visualiser la répartition des mentions par pays
    resultat = pd.pivot_table(pays_mention, values='count', index='country', columns='mention', aggfunc='sum')
    st.info("**Voici le tableau des résultats :**")
    st.write(resultat)


    # Question 3
    st.write("**3. Quel est le pourcentage d'étudiants ayant une note en dessous de la moyenne "
             "en 'data_deployement' parmi ceux qui ont échoué ?**")

    # Pourcentage d'échec à l'évaluation de "data_deployement" parmi ceux qui ont échoué
    failed_deployement_percentage = (df[df['admis'] == 'non']['data_deployement'] < 50).mean() * 100
    st.write("Le pourcentage d'échec à l'évaluation de 'data_deployement' parmi ceux qui ont échoué :", failed_deployement_percentage, "%")


    # Question 4
    st.write("**4. Quelle est la moyenne des notes de 'python' pour les étudiants ayant obtenu une mention 'assez bien' en 2022 ?**")

    # Moyenne des notes de "python" pour les étudiants ayant obtenu une mention "bien" en 2022
    mean_python_assez_bien_2022 = df[(df['mention'] == 'bien') & (df['annee'] == 2022)]['python'].mean()
    st.write("La moyenne des notes de 'python' pour les étudiants ayant obtenu une mention 'bien' en 2022 :", mean_python_assez_bien_2022)


    # Question 5
    st.write("**5. Quelle sont les matières dans lesquelles il faut beaucoup travailler pour réussir facilement**")

    # Sélectionner uniquement les colonnes numériques pour le calcul de corrélation
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])

    # Calculer la corrélation de chaque colonne avec la colonne "moyenne"
    correlation_with_mean = numeric_columns.corr()['moyenne'].drop('moyenne')
    st.write("Voici un tableau du coef de correlation de ces colonnes avec la moyenne")
    st.write(correlation_with_mean)


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
afficher_driven_decision()

