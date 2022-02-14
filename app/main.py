from flask import Flask, render_template, request, redirect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime


sns.set_theme()
app = Flask(__name__)


@app.route("/")
def index():
    train_data = pickle.load(open("train_data.pkl", "rb"))
    print(train_data)

    return render_template("home.html",
                           importance="importance.png",
                           heat="heat.png",
                           top2="top2.png",
                           confusion="confusion.png",
                           accuracy=train_data['accuracy'],
                           date=train_data['date'],
                           test=train_data['test'])


@app.route("/retrain")
def retrain():
    df = pd.read_csv("app/data/mushrooms.csv", sep=",")
    cleaned_df = df.dropna()
    test_size = 0.8
    X = cleaned_df.iloc[:, 1:25].applymap(lambda x: ord(x))
    Y = cleaned_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    accuracy_percentage = accuracy_score(clf.predict(X_test), y_test)
    feature_importance = clf.feature_importances_
    save_plots(df, feature_importance, clf, X_test, y_test)
    train_data = {'accuracy': accuracy_percentage,
                  'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                  'test': test_size}
    pickle.dump(train_data, open("train_data.pkl", "wb"))
    pickle.dump(clf, open("shrooming.pkl", "wb"))
    return redirect("/")


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


def save_plots(df, feature_importance, clf, X_test, y_test):
    importance_dict = gen_importance_dict(df, feature_importance)
    return {
        "importance": save_importance_plot(importance_dict),
        "corr_heat": save_curr_heat_plot(df),
        "top2": top2(df, importance_dict),
        "confusion": save_confusion_matrix(clf, X_test, y_test)
    }


def save_confusion_matrix(clf, X_test, y_test):
    output_path = "confusion.png"
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig('app/static/' + output_path, bbox_inches='tight', dpi=100)
    plt.close()


def gen_importance_dict(df, feature_importance):
    shroom_attributes = df.columns.tolist()
    shroom_attributes.pop(0)
    shroom_dict = {shroom_attributes[i]: feature_importance.tolist()[i]
                   for i in range(len(shroom_attributes))}
    return shroom_dict


def save_curr_heat_plot(df):
    output_path = "heat.png"
    sns.heatmap(df.applymap(lambda x: ord(x)).corr())
    plt.title("Correlation Heatmap of Attributes")
    plt.savefig('app/static/' + output_path, bbox_inches='tight', dpi=100)
    plt.close()
    return output_path


def save_importance_plot(importance):
    output_path = 'importance.png'
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    keys = importance.keys()
    values = importance.values()

    ax.bar(keys, values)
    plt.xticks(rotation=90)
    plt.ylabel("Importance Percentage")
    plt.title("Feature Importance of Trained Model")
    plt.savefig('app/static/' + output_path, bbox_inches='tight', dpi=100)
    plt.close()
    return output_path


def top2(df, importance_dict):
    output_path = 'top2.png'

    top_4_attr = sorted(
        importance_dict, key=importance_dict.get, reverse=True)[0:4]

    top1 = top_4_attr[0]
    top2 = top_4_attr[1]
    sub_df = df[[top1, top2]].applymap(lambda x: ord(x))

    kmeans = KMeans(n_clusters=2).fit(sub_df)
    centroids = kmeans.cluster_centers_

    plt.scatter(sub_df[top1], sub_df[top2], c=kmeans.labels_.astype(
        float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300)
    plt.ylabel(top2)
    plt.xlabel(top1)

    xticks = sub_df[top1].unique().tolist()
    xlabels = map(lambda x: chr(x), xticks)
    plt.xticks(xticks, xlabels)

    yticks = sub_df[top2].unique().tolist()
    ylabels = map(lambda y: chr(y), yticks)
    plt.yticks(yticks, ylabels)
    plt.title("K-means Clustering of the 2 Top Important Attributes")
    plt.savefig('app/static/' + output_path, bbox_inches='tight', dpi=100)
    plt.close()
    return output_path
