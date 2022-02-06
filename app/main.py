from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    df = pd.read_csv("app/data/mushrooms.csv", sep=",")
    cleaned_df = df.dropna()
    X = cleaned_df.iloc[:, 1:25].applymap(lambda x: ord(x))
    Y = cleaned_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.7)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # print(clf.feature_importances_)
    pickle.dump(clf, open("shrooming.pkl", "wb"))

    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    cap_shape = request.form['cap_shape']
    cap_surface = request.form["cap_surface"]
    cap_color = request.form["cap_color"]
    bruises = request.form["bruises"]
    odor = request.form["odor"]
    gill_attachment = request.form["gill_attachment"]
    gill_spacing = request.form["gill_spacing"]
    gill_size = request.form["gill_size"]
    gill_color = request.form["gill_color"]
    stalk_shape = request.form["stalk_shape"]
    stalk_root = request.form["stalk_root"]
    stalk_surface = request.form["stalk_surface"]
    stalk_surface_below_ring = request.form["stalk_surface_below_ring"]
    stalk_color_above_ring = request.form["stalk_color_above_ring"]
    stalk_color_below_ring = request.form["stalk_color_below_ring"]
    veil_type = request.form["veil_type"]
    veil_color = request.form["veil_color"]
    ring_number = request.form["ring_number"]
    ring_type = request.form["ring_type"]
    spore_print = request.form["spore_print"]
    population = request.form["population"]
    habitat = request.form["habitat"]

    model = pickle.load(open("shrooming.pkl", "rb"))
    dataframe_for_prediction = pd.DataFrame([[cap_shape,
                                              cap_surface,
                                              cap_color,
                                              bruises,
                                              odor,
                                              gill_attachment,
                                              gill_spacing,
                                              gill_size,
                                              gill_color,
                                              stalk_shape,
                                              stalk_root,
                                              stalk_surface,
                                              stalk_surface_below_ring,
                                              stalk_color_above_ring,
                                              stalk_color_below_ring,
                                              veil_type,
                                              veil_color,
                                              ring_number,
                                              ring_type,
                                              spore_print,
                                              population,
                                              habitat]]).applymap(lambda x: ord(x))
    prediction = model.predict(dataframe_for_prediction)

    if prediction == ['e']:
        result = "Edible"
        image = "edible.jpeg"
    else:
        result = "Poisonous"
        image = "poisonous.jpeg"

    return render_template("result.html", result=result, image=image)
