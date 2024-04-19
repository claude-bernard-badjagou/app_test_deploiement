import streamlit as st
from PIL import Image


def afficher_accueil():

    #st.set_page_config(
    #    page_title = "Multipage App",
    #    page_icon = "🤩"
    #)

    st.title("Les résultats de mon école")

    image = Image.open("Gomycode.png")
    st.image(image, caption="Image de gomycode sur twitter", use_column_width=True)



    # Bienvenue sur l'application de suivi des étudiants en data science

    ## À propos de l'application
    st.write("""
    <div style="text-align: justify">
    Cette application a été conçue pour suivre les performances des étudiants
    en data science. Leurs performances dans différents domaines depuis la création de l'école
    sont analysées pour fournir des informations précieuses pour la prise de décision.
    De plus des modèles sont crés pour prédire ou classiffier les informations sur les etudiants.
    </div>
    """, unsafe_allow_html=True)
    st.write("NB : Les données ne sont pas de vraies données mais des données générées.")

    # Titre
    st.header("Fonctionnalités de l'application")

    # Exploration des données
    st.subheader("Exploration des données")
    st.write("Analyse des données des étudiants pour obtenir des insights précieux.")

    # Manipulation des données
    st.subheader("Manipulation des données")
    st.write("Filtrage, triage et modification des données selon le besoins.")

    # Modélisation des données
    st.subheader("Modélisation des données")
    st.write("Modèles d'apprentissage automatique et de d'apprentissage profond pour prédire ou classifier les résultats des étudiants.")

    # Visualisation des données
    st.subheader("Visualisation des données")
    st.write("Visualisation des données à l'aide de graphiques interactifs pour une compréhension plus claire.")

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
afficher_accueil()