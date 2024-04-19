import streamlit as st
import pickle as pk


def afficher_about_us():
    # Titre de la page

    st.title('A prospos de nous')

    def apropos():
        # Définition du style CSS pour justifier le texte
        st.markdown("""
            <style>
            /* Définition du style pour justifier le texte */
            .css-1aumxhk {
                text-align: justify;
            }
            </style>
            """, unsafe_allow_html=True)

        st.write("Bienvenue sur **SCHOOL PERFORM** !")
        st.write("""Nous sommes une équipe passionnée par la data science et son potentiel infini
                 pour transformer les données en insights puissants. Notre équipe est composée de 
                 [nombre] professionnels chevronnés de la data science, chacun apportant sa propre 
                 expertise et sa passion unique à chaque projet.""")

        st.header("Notre Vision")
        st.write("""Notre vision est de façonner un avenir où les données deviennent le moteur de "
                 l'innovation et de la prise de décision dans tous les secteurs. Nous croyons fermement
                  que la data science a le pouvoir de résoudre certains des défis les plus pressants de 
                  notre époque, qu'il s'agisse de problèmes commerciaux complexes, de questions sociales ou 
                  de défis environnementaux. Nous aspirons à créer un monde où chaque décision est  éclairée 
                  par des données fiables et des analyses approfondies, permettant ainsi aux entreprises et 
                  aux organisations de prospérer dans un environnement en constante évolution.""")

        st.header("Notre Mission")
        st.write("""Notre mission est de démocratiser l'accès à l'analyse de données avancée et de rendre 
                 les insights compréhensibles et exploitables pour tous. Nous croyons fermement que la 
                 data science devrait être accessible à tous, quelle que soit leur expertise technique, 
                 et c'est pourquoi nous nous efforçons de créer des outils intuitifs et conviviaux pour 
                 transformer les données brutes en informations exploitables.""")

        st.header("Notre Passion")
        st.write("""Ce qui nous anime, c'est notre passion pour les données. Nous adorons explorer de 
                 nouveaux ensembles de données, découvrir des tendances cachées et raconter des histoires
                 captivantes à travers les données. Chaque projet est une nouvelle aventure pour nous, 
                 et nous sommes toujours excités à l'idée d'explorer les possibilités infinies que 
                 les données offrent.""")

        st.header("Pourquoi Nous Choisir")
        st.write("Lorsque vous travaillez avec nous, vous bénéficiez de l'expertise d'une équipe dévouée et passionnée, prête à relever les défis les plus complexes en matière de data science. Nous sommes à l'écoute de vos besoins et nous nous engageons à fournir des solutions personnalisées et innovantes qui répondent à vos objectifs commerciaux. Que vous soyez une start-up ambitieuse ou une entreprise établie, nous sommes là pour vous aider à libérer le potentiel de vos données et à prendre des décisions éclairées pour l'avenir de votre entreprise.")

        st.write("[Nous contacter](http://localhost:8501/Nous_contacter)")
        st.write("**Adresse e-mail :** ibadjagou@gmail.com")
        st.markdown("**Téléphone :** [+225 05 54 47 30 01](tel:+2250554473001)")
        st.write("**Localisation :** [Cocody, Rond Point Palméraie]")

    apropos()

afficher_about_us()
