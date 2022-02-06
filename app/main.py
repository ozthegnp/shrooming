from flask import Flask, render_template, request, Response
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import pickle
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import io
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)


@app.route("/")
def index():
    df = pd.read_csv("app/data/mushrooms.csv", sep=",")
    cleaned_df = df.dropna()

    X = cleaned_df.iloc[:, 1:25].applymap(lambda x: ord(x))
    Y = cleaned_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    accuracy_percentage = accuracy_score(clf.predict(X_test), y_test)
    feature_importance = clf.feature_importances_
    # plot_confusion_matrix(clf, X_test, y_test)
    plots = save_plots(df, feature_importance, accuracy_percentage)

    pickle.dump(clf, open("shrooming.pkl", "wb"))
    return render_template("home.html", importance=plots["importance"])


@ app.route("/predict", methods=["GET", "POST"])
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
    proba = model.predict_proba(dataframe_for_prediction)
    print("proba")
    print(proba)
    df = pd.read_csv("app/data/mushrooms.csv", sep=",")
    cleaned_df = df.dropna()
    X = cleaned_df.iloc[:, 1:25].applymap(lambda x: ord(x))

    if prediction == ['p']:
        result = "Edible"
        image = "edible.jpeg"
    else:
        result = "Poisonous"
        image = "poisonous.jpeg"
    return render_template("result.html",
                           result=result,
                           image=image,
                           probability=round(max(proba[0]), 2))


def save_plots(df, feature_importance, accuracy_percentage):
    importance_dict = gen_importance_dict(df, feature_importance)
    print(importance_dict)
    return {
        "importance": save_importance_plot(importance_dict),
        "top_corr": save_top_corr_plot(df, importance_dict),
        # "year": 1964
    }


def gen_importance_dict(df, feature_importance):
    shroom_attributes = df.columns.tolist()
    shroom_attributes.pop(0)
    shroom_dict = {shroom_attributes[i]: feature_importance.tolist()[i]
                   for i in range(len(shroom_attributes))}
    return shroom_dict


def save_top_corr_plot(df, importance_dict):
    top_4_attr = sorted(
        importance_dict, key=importance_dict.get, reverse=True)[0:4]
    print(top_4_attr)
    return


def save_importance_plot(importance):
    output_path = 'importance.png'
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    keys = importance.keys()
    values = importance.values()

    ax.bar(keys, values)
    plt.xticks(rotation=90)
    plt.ylabel("percentage")
    plt.title("Feature Importance")
    plt.savefig('app/static/' + output_path, bbox_inches='tight', dpi=100)
    plt.close()
    return output_path
