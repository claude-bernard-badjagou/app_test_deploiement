import streamlit as st


def afficher_nous_contacter():
    # Titre de la page
    st.title("Nous contacter")


    st.subheader("Entreprise, particulier, data scientist et autres")
    st.write("""Vous avez un projet de data science en tête ou vous souhaitez simplement 
    en savoir plus sur ce que nous pouvons faire avec vous ? N'hésitez pas à nous contacter 
    dès aujourd'hui pour une consultation gratuite. Nous sommes impatients de discuter avec 
    vous et de transformer vos idées en réalité.""")


    # Aller copier ce code dans formsubmit
    formulaire_de_contact = """
    <form action="https://formsubmit.co/ibadjagou@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder = "Votre nom" required>
         <input type="email" name="email" placeholder = "Votre email" required>
         <textarea name="message" placeholder="Ecrivez votre message ici"></textarea>
         <button type="submit">Send</button>
    </form>"""
    st.markdown(formulaire_de_contact, unsafe_allow_html=True)


    # Mise en forme avec du css
    # Utiliser le fichier css
    def fichier_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</tyle>", unsafe_allow_html=True)

    fichier_css("pages/style.css")

afficher_nous_contacter()
