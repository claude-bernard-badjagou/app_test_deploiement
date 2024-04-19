import streamlit as st
import pickle as pk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
def afficher_data_modelisation():
    # Warnings
    st.set_option('deprecation.showPyplotGlobalUse', False)


    # Titre de la page
    st.title("Data modelisation")


    # Chargement des données
    df = pk.load(open("data_preprocessed.pkl", "rb"))


    # Ajout de la colonne "admis_numeric" et mention_numérique
    df['admis_numeric'] = df['admis'].map({'oui': 1, 'non': 0})
    df['mention_numeric'] = df['mention'].map({'refusée': 0, 'passable': 1,
                                               'assez bien': 2, 'bien': 3, 'très bien': 4, 'excellent': 5})

    # Créer un bouton pour afficher un message d'information
    if st.checkbox("**Cliquez ici pour masquer l'information**", value=True):
        # display the text if the checkbox returns True value
        st.write("""**Le résultat du modèle choisi sera affiché en bas de cette information** \n
        Sur cette page vous pouvez faire de la prédiction . Ainsi, lorsque vous exécutez ce \n
        code, le texte sera affiché par défaut car la case à cocher sera cochée initialement. 
        L'utilisateur peut ensuite décocher la case à cocher pour masquer le texte.
        """)


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
            y = dataframe["admis"]

        elif model == "Random Forest":
            x = dataframe[["moyenne"]]
            y = dataframe["mention"]

        elif model == "Support Vector Machin":
            x = dataframe[["moyenne"]]
            y = dataframe["mention"]

        elif model == "ANN ou DNN":
            x = dataframe[
                ['sql', 'excel', 'maths', 'python', 'poo', 'tableau', 'data_exploration', 'data_manipulation', 'data_viz',
                 'data_transformation', 'data_modelisation', 'data_deployement', 'pack_office', 'result_presentation']]
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


    #Afficher les étiquettes sur les matrices de confusion et autre
    classe_names = ["non admis", "admis"]


    # ✏️ Afficher les graphiques de performance sans try et après avec try except
    def plot_perf(graphes):
        if "Confucsion matrix" in graphes:
            st.subheader("Matrice de confusion")
            try:
                ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
            except Exception as e:
                st.warning(f"La Matrice de confusion ne peut pas être représenté avec les données du modèle: {str(e)}")

        if "ROC Curve" in graphes:
            st.subheader("La courbe ROC (Receiver Operating Characteristic)")
            try:
                RocCurveDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
                st.info("Une courbe ROC idéale se rapproche du coin supérieur gauche du graphique, "
                        "ce qui indique un modèle avec une sensibilité élevée et un faible taux de faux positifs.")
            except Exception as e:
                st.warning(f"La courbe ROC ne peut pas être représenté avec les données du modèle: {str(e)}")

        if "Precision_Recall Curve" in graphes:
            st.subheader("La courbe de Précision Recall)")
            try:
                PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
                st.info("La courbe PR met l'accent sur la capacité du modèle à bien classer les échantillons positifs, "
                        "ce qui est important dans les cas où les classes sont très déséquilibrées. ")
                st.write("""Une AUC-PR proche de 1 indique un modèle idéal, 
                où chaque prédiction positive est correcte et chaque prédiction négative est incorrecte. 
                Proximité du coin supérieur droit : Comme pour la courbe ROC, une courbe PR idéale tend 
                à se rapprocher du coin supérieur droit du graphique. Cela signifie que le modèle atteint 
                à la fois une précision élevée et un rappel élevé pour un seuil de classification donné.
                Croissance rapide de la courbe : Une courbe PR idéale augmente rapidement à partir de 
                l'origine, ce qui signifie qu'elle atteint une haute précision pour un rappel relativement faible. 
                Cela indique que le modèle est capable de bien classer les échantillons positifs 
                dès le début de la prédiction. Pas de "dents de scie" : Une courbe PR idéale est lisse, 
                sans "dents de scie" ou de fortes variations. Cela signifie que le modèle maintient 
                une précision élevée même lorsqu'il rappelle un grand nombre d'échantillons positifs.
                """)
            except Exception as e:
                st.warning(f"La courbe de Précision Recall ne peut pas être représenté avec les données du modèle: {str(e)}")

    # Réglage du paramètres de chaque modéle

    # 1️⃣ Regression Linéaire simple
    if model == "Regression Linéaire simple":
        if st.sidebar.button("Prédire", key="regression"):
            st.subheader("Résultat de la Regression Linéaire simple")

            # Initialiser le model
            model = LinearRegression()
            # Entrainer le model
            model.fit(x_train, y_train)
            # Prédiction du model
            y_pred = model.predict(x_test)
            # Calcul du MSE pour évaluer la performance du modèle
            mse = mean_squared_error(y_test, y_pred)
            # Calcul du R carré
            r2 = r2_score(y_test, y_pred)
            st.write("l'Erreur quadratique moyenne (MSE) du model est :", mse)
            st.write("Le R carré du model est :", r2)

            if r2 < 0.8:
                st.write("Il faut normalement améliorer ce modèle avant de l'utiliser pour faire des prédictions")
                st.error("VOULEZ-VOUS QUAND MÊME FAIRE DES PRÉDICTIONS AVEC CE MODÈLE ?")
                # lien de la page de prédiction
                st.write(" [Je veux faire des prédictions](http://localhost:8501/Model_Prediction)")

            else:
                st.success("BON MODÈLE ! LE R CARRÉ EST SUPÉRIEUR OU ÉGAl À 0.8 ET PROCHE DE 1")
                st.info("VOUS POUVEZ PRÉDIRE LA NOTE DE DATA_EXPLORATION AVEC CE MODÈLE")
                # lien de la page de prédiction
                st.write(" [Je veux faire des prédictions](http://localhost:8501/Model_Prediction)")

    # 2️⃣ Regression Linéaire multiple
    elif model == "Regression Linéaire multiple":
        if st.sidebar.button("Prédire", key="regression"):
            st.subheader("Résultat de la Regression Linéaire multiple")

            # Initialiser le model
            model = LinearRegression()
            # Entrainer le model
            model.fit(x_train, y_train)
            # Prédiction du model
            y_pred = model.predict(x_test)
            # Calcul du MSE pour évaluer la performance du modèle
            mse = mean_squared_error(y_test, y_pred)
            # Calcul du R carré
            r2 = r2_score(y_test, y_pred)
            st.write("l'Erreur quadratique moyenne (MSE) du model est :", mse)
            st.write("Le R carré du model est :", r2)
            st.info("Plus le R carré est proche de 1 plus le model fait de bonne prédiction")

    # 3️⃣ Regression logistique
    elif model == "Regression logistique":
        st.sidebar.subheader("Les hyperparamètres du modéle")

        hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

        n_max_iter = st.sidebar.number_input("Choisir le nombre maximal d'itération", 100, 1000, step = 10)

        graphes_perf = st.sidebar.multiselect("Choisir un ou des graphiques de performance du model à afficher",
                                              ("Confucsion matrix", "ROC Curve", "Precision_Recall Curve"))

        if st.sidebar.button("Prédire", key="logistic_regression"):
            st.subheader("Résultat de la Regression logistique")

            # Initialiser le modèle
            model = LogisticRegression(C=hyp_c, max_iter=n_max_iter, random_state=seed)

            # Entrainer le modèle
            model.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = model.predict(x_test)

            # Calcul des metrics de performances

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label="oui")
            recall = recall_score(y_test, y_pred, pos_label="oui")

            #Afficher les métrics
            st.write("Exactitude du modèle :", accuracy)
            st.write("Précision du modèle :", precision)
            st.write("Recall du modèle :", recall)

            # Afficher les graphiques de performances
            plot_perf(graphes_perf)


    # 4️⃣ Random Forest
    elif model == "Random Forest":
        st.sidebar.subheader("Hyperparameters for the Random Forest model")

        n_estimators = st.sidebar.number_input("Nombre d'estimateurs", 1, 1000, step=1)
        max_depth = st.sidebar.slider("Max depth of each tree", 1, 20, 10)

        graphes_perf = st.sidebar.multiselect("Select one or more performance graphs to display",
                                              ["Confusion matrix", "ROC Curve", "Precision_Recall Curve"])

        if st.sidebar.button("Predict", key="random_forest"):
            st.subheader("Random Forest Model Results")

            # Initialize the Random Forest model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)

            # Train the model
            model.fit(x_train, y_train)

            # Make predictions
            y_pred = model.predict(x_test)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')

            # Display metrics
            st.write("Model Accuracy:", accuracy)
            st.write("Model Precision:", precision)
            st.write("Model Recall:", recall)



    # 5️⃣ Support Vector Machin
    elif model == "Support Vector Machin":
        st.sidebar.subheader("Les hyperparamètres du modéle")

        hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

        kernel = st.sidebar.radio("Choisir le noyau", ("rbf", "linear", "poly", "sigmoid"))

        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))

        graphes_perf = "Confucsion matrix"

        if st.sidebar.button("Prédire", key="classifivation multiclasse"):
            st.subheader("Résultat de Support Vecteur Machine (SVM)")

            # Initialiser le modèle svc pour la classification
            model = SVC(C=hyp_c, kernel = kernel, gamma = gamma, decision_function_shape='ovo')

            # Entrainer le modèle
            model.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = model.predict(x_test)

            # Calcul des metrics de performances

            #accuracy = accuracy_score(y_test, y_pred)
            #precision = precision_score(y_test, y_pred)
            #recall = recall_score(y_test, y_pred)

            #Afficher les métrics
            #st.write("Exactitude du modèle :", accuracy)
            #st.write("Précision du modèle :", precision)
            #st.write("Recall du modèle :", recall)

            # Afficher les graphiques de performances
            plot_perf(graphes_perf)
            st.write()

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

                # Affichage des graphiques de performance
            plot_perf(graphes_perf)

afficher_data_modelisation()