import streamlit as st
import pickle as pk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score

def afficher_model_prediction():
    # Warnings
    st.set_option('deprecation.showPyplotGlobalUse', False)


    # Titre de la page
    st.title("Modèle Prédiction")


    # Chargement des données
    df = pk.load(open("data_preprocessed.pkl", "rb"))


    # Ajout de la colonne "admis_numeric" et mention_numérique
    df['admis_numeric'] = df['admis'].map({'oui': 1, 'non': 0})
    df['mention_numeric'] = df['mention'].map({'refusée': 0, 'passable': 1,
                                               'assez bien': 2, 'bien': 3, 'très bien': 4, 'excellent': 5})

    # Créer un bouton pour afficher un message d'information
    if st.checkbox("**Cliquez ici pour masquer l'information**", value=True):
        # display the text if the checkbox returns True value
        st.write("**Le résultat du modèle choisi sera affiché en bas de cette information**")


    # Les modèles disponibles
    model = st.sidebar.selectbox("Choisissez un model",
                                 ["Regression Linéaire simple", "Regression Linéaire multiple",
                                  "Regression logistique", "Random Forest", "Support Vector Machin", "ANN ou DNN"])


    # ✂️ Selection et découpage des données
    seed = 123
    def select_split(dataframe):

        if model == "Regression Linéaire simple":
            x = dataframe[["maths"]]
            y = dataframe["data_exploration"]

        elif model == "Regression Linéaire multiple":
            x = dataframe[['sql', 'excel', 'maths', 'python', 'poo', 'tableau', 'data_exploration','data_manipulation', 'data_viz', 'data_transformation', 'data_modelisation','data_deployement', 'pack_office', 'result_presentation']]
            y = dataframe["moyenne"]

        elif model == "Regression logistique":
            x = dataframe[["moyenne"]]
            y = dataframe["admis_numeric"]

        elif model == "Random Forest":
            x = dataframe[["moyenne"]]
            y = dataframe["mention"]

        elif model == "Support Vector Machin":
            x = dataframe[["moyenne"]]
            y = dataframe["mention"]


        elif model == "ANN ou DNN":

            x = dataframe[['sql', 'excel', 'maths', 'python', 'poo', 'tableau',
                           'data_exploration', 'data_manipulation', 'data_viz',
                           'data_transformation', 'data_modelisation', 'data_deployement',
                           'pack_office', 'result_presentation']]
            y = dataframe["mention_numeric"]

        else:
            # Définir un comportement par défaut si nécessaire
            x = dataframe[["maths"]]
            y = dataframe["data_exploration"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test


    # Création des variables d'entrainement et test
    x_train, x_test, y_train, y_test = select_split(dataframe=df)
    # Conversion des séries pandas en tableaux bidimensionnels
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    x_train, x_test, y_train, y_test = select_split(dataframe=df)


    # Réglage du paramètres de chaque modéle

    # 1️⃣ Regression Linéaire simple
    if model == "Regression Linéaire simple":
        # Demander à l'utilisateur de saisir la valeur de la caractéristique
        maths_data = st.sidebar.number_input("Entrez la note en maths:", min_value=0.0, max_value=100.0,
                                     value=0.0)

        if st.sidebar.button("Prédire la note de data exploration", key="regression"):
            st.subheader("Prédiction de la note de data exploration avec la Regression Linéaire simple")
            # Initialiser le model
            model = LinearRegression()
            # Entrainer le model
            model.fit(x_train, y_train)

            data_explo = model.predict([[maths_data]])
            # Afficher le résultat de la prédiction
            st.write(f"Pour une note {maths_data} en maths, la note de data exploration est : {data_explo}")



    # 2️⃣ Regression Linéaire multiple
    elif model == "Regression Linéaire multiple":

        # Demander à l'utilisateur de saisir les valeurs des caractéristiques
        note = {}
        note['sql'] = st.sidebar.slider("Note en SQL :", min_value=0.0, max_value=100.0, value=0.0)
        note['excel'] = st.sidebar.slider("Note en Excel :", min_value=0.0, max_value=100.0, value=0.0)
        note['maths'] = st.sidebar.slider("Note en Mathématiques :", min_value=0.0, max_value=100.0, value=0.0)
        note['python'] = st.sidebar.slider("Note en Python :", min_value=0.0, max_value=100.0, value=0.0)
        note['poo'] = st.sidebar.slider("Note en Programmation Orientée Objet (POO) :", min_value=0.0, max_value=100.0,
                                        value=0.0)
        note['tableau'] = st.sidebar.slider("Note en Tableau de Bord :", min_value=0.0, max_value=100.0, value=0.0)
        note['data_exploration'] = st.sidebar.slider("Note en Exploration des Données :", min_value=0.0,
                                                     max_value=100.0, value=0.0)
        note['data_manipulation'] = st.sidebar.slider("Note en Manipulation des Données :", min_value=0.0,
                                                      max_value=100.0, value=0.0)
        note['data_viz'] = st.sidebar.slider("Note en Visualisation de Données :", min_value=0.0, max_value=100.0,
                                             value=0.0)
        note['data_transformation'] = st.sidebar.slider("Note en Transformation des Données :", min_value=0.0,
                                                        max_value=100.0, value=0.0)
        note['data_modelisation'] = st.sidebar.slider("Note en Modélisation des Données :", min_value=0.0,
                                                      max_value=100.0, value=0.0)
        note['data_deployement'] = st.sidebar.slider("Note en Déploiement des Données :", min_value=0.0,
                                                     max_value=100.0, value=0.0)
        note['pack_office'] = st.sidebar.slider("Note en Pack Office :", min_value=0.0, max_value=100.0, value=0.0)
        note['result_presentation'] = st.sidebar.slider("Note en Présentation des Résultats :", min_value=0.0,
                                                        max_value=100.0, value=0.0)

        if st.sidebar.button("Prédire la moyenne", key="regression"):
            st.subheader("Prédiction de la **moyenne** avec la Regression Linéaire multiple")

            # Initialiser le model
            model = LinearRegression()
            # Entrainer le model
            model.fit(x_train, y_train)

            # Convertir les valeurs des caractéristiques en un tableau 2D
            notes = [[note['sql'], note['excel'], note['maths'], note['python'],
                               note['poo'], note['tableau'], note['data_exploration'],
                               note['data_manipulation'], note['data_viz'], note['data_transformation'],
                               note['data_modelisation'], note['data_deployement'], note['pack_office'],
                               note['result_presentation']]]
            # Prédiction du model
            moyenne_pred = model.predict(notes)
            st.write(f"Pour ces notes dans les différentes matières, : {moyenne_pred} serait obtenue comme moyenne")


    # 3️⃣ Regression logistique
    elif model == "Regression logistique":
        # Demander à l'utilisateur de saisir les valeurs des caractéristiques
        features = {}
        features['moyenne'] = st.sidebar.slider("Entrez la valeur de Moyenne :", min_value=0.0, max_value=100.0, value=0.0)

        if st.sidebar.button("Prédire si admissible", key="logistic_regression"):
            st.subheader("Prédiction de l'admissibilité avec la Regression logistique")
            # Initialiser le model
            model = LinearRegression()
            # Entrainer le model
            model.fit(x_train, y_train)

            moyenne_input = [[features['moyenne']]]
            # Effectuer la prédiction avec le modèle de régression logistique
            admissible = model.predict(moyenne_input)

            # Afficher le résultat de la prédiction
            if admissible < 0.5:
                decision = "**n'est pas admissible**"
            else:
                decision = "**est admissible**"

            st.write(f"Cette moyenne **{features['moyenne']} sur 100** {decision}")


    # 4️⃣ Random Forest
    elif model == "Random Forest":
        st.sidebar.subheader("Hyperparameters for the Random Forest model")

        n_estimators = st.sidebar.number_input("Nombre d'estimateurs", 1, 1000, step=1)
        max_depth = st.sidebar.slider("Max depth of each tree", 1, 20, 10)
        # Demander à l'utilisateur de saisir les valeurs des caractéristiques
        features = {}
        features['moyenne'] = st.sidebar.slider("Moyenne calculée (optionel):", min_value=0.0, max_value=100.0, value=0.0)
        moyenne_input_opt = [[features['moyenne']]]


        if st.sidebar.button("Predire la mention", key="random_forest"):
            st.subheader("Prédirte la mention avec du Random Forest")

            # Initialize the Random Forest model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)

            # Train the model
            model.fit(x_train, y_train)

            # Make predictions
            y_pred = model.predict(x_test)
            mention_pred = model.predict(moyenne_input_opt)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')

            # Display metrics
            st.write("Model Accuracy:", accuracy)
            st.write("Model Precision:", precision)
            st.write("Model Recall:", recall)

            # Afficher le résultat de la prédiction
            st.write(f"Pour cette moyenne **{features['moyenne']}**, la mention prédite est : {mention_pred}")


    # 5️⃣ Support Vector Machin
    elif model == "Support Vector Machin":
        st.sidebar.subheader("Les hyperparamètres du modéle")

        hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

        kernel = st.sidebar.radio("Choisir le noyau", ("rbf", "linear", "poly", "sigmoid"))

        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))

        # Demander à l'utilisateur de saisir les valeurs des caractéristiques
        notes = {}
        notes['moyenne'] = st.sidebar.slider("Moyenne calculée (optionel):", min_value=0.0, max_value=100.0, value=0.0)
        moyenne_input_option = [[notes['moyenne']]]

        if st.sidebar.button("Prédire la mention", key="classifivation multiclasse"):
            st.subheader("Prédire la mention avec le Support Vecteur Machine (SVM)")

            # Initialiser le modèle svc pour la classification
            model = SVC(C=hyp_c, kernel = kernel, gamma = gamma, decision_function_shape='ovo')

            # Entrainer le modèle
            model.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = model.predict(x_test)
            mention_predict = model.predict(moyenne_input_option)

            # Afficher le résultat de la prédiction
            st.write(f"Pour cette moyenne **{notes['moyenne']}**, la mention prédite est : {mention_predict}")

            # Calcul des metrics de performances

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')

            # Afficher les métrics
            st.write("Exactitude du modèle :", accuracy)
            st.write("Précision du modèle :", precision)


    # 6️⃣ ANN ou DNN
    elif model == "ANN ou DNN":
        # Demander les paramètres du model
        # Demander le nombre de couches cachés
        n_hiden_layer = st.sidebar.number_input("Nombre de couche cachée", 1,10,1)

        # Demander le nombre de neuronne par couche caché
        n_neurones_layer = [st.sidebar.number_input(f"Nombre de neurones dans la couche {i + 1}", 50, 1000, 100) for i in range(n_hiden_layer)]

        # Demander la fonction dactivation de chaque couche caché
        activation_functions = []
        for i in range(n_hiden_layer):
            activation_function = st.sidebar.selectbox(f"Choisir la fonction d'activation de la couche {i + 1}",
                                                       ["relu", "sigmoid", "tanh", "softmax"])
            activation_functions.append(activation_function)

        # Demander la fonction de perte
        loss_function = st.sidebar.selectbox("Fonction de perte",
                            ["binary_crossentropy", "categorical_crossentropy", "Mean Squared Error (MSE)"])

        # Demander l'optimiseur
        optimiseur = st.sidebar.selectbox("Choisisir l'optimiseur", ["adam", "sgd", "rmsprop"])

        # Métriques à surveiller
        metrics = st.sidebar.multiselect("Métriques à surveiller", ["accuracy", "precision", "recall"])

        # Configurer early stopping pour éviter le suradjustement
        early_stopping = st.sidebar.checkbox("Early stopping")
        if early_stopping:
            monitor = st.sidebar.selectbox("Choisir le moniteur pour early stopping", ["val_loss", "val_accuracy"])
            patience = st.sidebar.number_input("Patience", 1, 10, 5)

        # Afficher les graphiques de performance
        graphes_perf = st.sidebar.multiselect("Choisir un ou des graphiques de performance du modèle à afficher",
                                              ["Confucsion matrix", "ROC Curve", "Precision_Recall Curve"])

        # Demander à l'utilisateur de saisir les valeurs des caractéristiques
        noted = {}
        noted['sql'] = st.sidebar.slider("Note en SQL :", min_value=0.0, max_value=100.0, value=0.0)
        noted['excel'] = st.sidebar.slider("Note en Excel :", min_value=0.0, max_value=100.0, value=0.0)
        noted['maths'] = st.sidebar.slider("Note en Mathématiques :", min_value=0.0, max_value=100.0, value=0.0)
        noted['python'] = st.sidebar.slider("Note en Python :", min_value=0.0, max_value=100.0, value=0.0)
        noted['poo'] = st.sidebar.slider("Note en Programmation Orientée Objet (POO) :", min_value=0.0, max_value=100.0,
                                        value=0.0)
        noted['tableau'] = st.sidebar.slider("Note en Tableau de Bord :", min_value=0.0, max_value=100.0, value=0.0)
        noted['data_exploration'] = st.sidebar.slider("Note en Exploration des Données :", min_value=0.0,
                                                     max_value=100.0, value=0.0)
        noted['data_manipulation'] = st.sidebar.slider("Note en Manipulation des Données :", min_value=0.0,
                                                      max_value=100.0, value=0.0)
        noted['data_viz'] = st.sidebar.slider("Note en Visualisation de Données :", min_value=0.0, max_value=100.0,
                                             value=0.0)
        noted['data_transformation'] = st.sidebar.slider("Note en Transformation des Données :", min_value=0.0,
                                                        max_value=100.0, value=0.0)
        noted['data_modelisation'] = st.sidebar.slider("Note en Modélisation des Données :", min_value=0.0,
                                                      max_value=100.0, value=0.0)
        noted['data_deployement'] = st.sidebar.slider("Note en Déploiement des Données :", min_value=0.0,
                                                     max_value=100.0, value=0.0)
        noted['pack_office'] = st.sidebar.slider("Note en Pack Office :", min_value=0.0, max_value=100.0, value=0.0)
        noted['result_presentation'] = st.sidebar.slider("Note en Présentation des Résultats :", min_value=0.0,
                                                        max_value=100.0, value=0.0)


        if st.sidebar.button("Prédire", key="classifivation multiclasse"):
            st.subheader("Résultat de ANN ou DNN")

            # Encodage des données cibles
            y_train = to_categorical(y_train, num_classes=6)
            y_test = to_categorical(y_test, num_classes=6)

            # Création du modèle ANN ou DNN
            model = Sequential()
            model.add(Dense(n_neurones_layer[0], input_dim=x_train.shape[1], activation=activation_functions[0]))
            for i in range(1, n_hiden_layer):
                model.add(Dense(n_neurones_layer[i], activation=activation_functions[i]))
            model.add(Dropout(st.sidebar.slider(f"Taux de dropout pour la couche {i + 1}", 0.0, 0.9, 0.1)))

            model.add(Dense(6, activation='softmax')) # Couche de sortie pour la classification binaire

            # Compilation du modèle
            model.compile(loss=loss_function, optimizer=optimiseur, metrics=metrics)

            early_stopping_callback = None
            # Entraînement du modèle
            callbacks = []
            if early_stopping:
                early_stopping_callback = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
                callbacks.append(early_stopping_callback)

                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2,
                      callbacks=callbacks)

            else:
                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

            # Prédiction du modèle
            y_pred = np.round(model.predict(x_test))

            # Convertir les valeurs des caractéristiques en un tableau 2D
            noted = np.array([[noted['sql'], noted['excel'], noted['maths'], noted['python'],
                      noted['poo'], noted['tableau'], noted['data_exploration'],
                      noted['data_manipulation'], noted['data_viz'], noted['data_transformation'],
                      noted['data_modelisation'], noted['data_deployement'], noted['pack_office'],
                      noted['result_presentation']]])
            # Prédiction du model
            mention_num_pred = model.predict(noted)

            # Obtenir l'indice de la classe prédite
            predicted_class_index = np.argmax(mention_num_pred)

            # Ici, vous devrez avoir une correspondance entre l'indice prédit et les étiquettes de départ
            # Par exemple, si votre modèle prédit un index de classe 0, alors vous pouvez définir une correspondance comme ceci :
            class_labels = ["refusée", "passable", "assez bien", "bien", "très bien", "excellent"]

            # Obtenir l'étiquette prédite en fonction de l'indice prédit
            predicted_label = class_labels[predicted_class_index]

            st.write(f"Pour ces notes dans les différentes matières, la mention serait: {predicted_label}")


            # Affichage des métriques
            # Essayer de calculer les métriques
            try:
                accuracy = accuracy_score(y_test, y_pred)
                st.write("Exactitude du modèle :", accuracy)
            except Exception as e_accuracy:
                st.warning(f"Impossible de calculer l'exactitude : {str(e_accuracy)}")

            try:
                precision = precision_score(y_test, y_pred, average='micro')
                st.write("Précision du modèle :", precision)
            except Exception as e_precision:
                st.warning(f"Impossible de calculer la précision : {str(e_precision)}")

            try:
                recall = recall_score(y_test, y_pred, average='micro')
                st.write("Recall du modèle :", recall)
            except Exception as e_recall:
                st.warning(f"Impossible de calculer le rappel : {str(e_recall)}")

afficher_model_prediction()
