import pygwalker as pyg
import pickle as pk
import streamlit.components.v1 as components
import streamlit as st

def afficher_make_visualisation():
    st.set_page_config(layout="wide")

    # Chargement des données prétraitées dans data manipulation
    df = pk.load(open("data_preprocessed.pkl", "rb"))


    # Ajouter un titre
    st.title("Faites vos visualisations ici")
    st.info("Glisser et déposer le nom des colonnes sous 'Field List' dans les axes x et axe y")


    # Générer le HTML en utilisant Pygwalker
    pyg_html = pyg.to_html(df)

    # Intégrer le HTML dans l'application Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)


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

afficher_make_visualisation()