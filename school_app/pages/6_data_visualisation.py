import streamlit as st
import pickle as pk
import plotly.express as px

def afficher_data_visualisation():
    # Titre de la page
    st.title("Data Visualization")

    # Chargement du fichier
    df = pk.load(open("data_preprocessed.pkl", "rb"))


    # Question 1: Quelle est la répartition des notes en maths parmi les étudiants ?
    st.subheader("1. Répartition des notes en maths")
    fig1 = px.histogram(df, x='maths', title="Répartition des notes en maths")
    fig1.update_layout(xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig1)

    # Question 2 : Distribution des notes en SQL et en Excel
    st.subheader("2. Distribution des notes en SQL et en Excel")
    fig2 = px.scatter(df, x='maths', y='data_exploration',
                      title="Distribution des notes en SQL et en Excel")
    fig2.update_layout(xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig2)

    # Question 3: Comment varie la moyenne des notes par pays d'origine des étudiants ?
    st.subheader("3. Variation de la moyenne des notes par pays d'origine")
    fig3 = px.bar(df, x='country', y='moyenne',
                  title="Variation de la moyenne des notes par pays d'origine")
    st.plotly_chart(fig3)


    # Question 4 : Répartition des notes en maths par pays
    st.subheader("4. Répartition des notes en maths par pays")
    fig4 = px.box(df, x='country', y='maths', title="Répartition des notes en maths par pays")
    st.plotly_chart(fig4)

    # Question 5 : Distribution des notes en Python parmi les étudiants admis et non admis
    st.subheader("5. Distribution des notes en Python parmi les étudiants admis et non admis")
    fig5 = px.histogram(df, x='python', color='admis',
                        title="Distribution des notes en Python parmi les étudiants admis et non admis")
    fig5.update_layout(xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig5)

    # Question 6 : Distribution des notes en moyenne parmi les étudiants admis et non admis
    st.subheader("6. Distribution des notes en moyenne parmi les étudiants admis et non admis")
    fig6 = px.scatter(df, x='moyenne', y='admis',
                      title="Distribution des notes en moyenne parmi les étudiants admis et non admis")
    fig6.update_layout(xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig6)

afficher_data_visualisation()

