import sklearn
from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def home():
    df = pd.read_csv("mushrooms.csv", sep=",")
    cleaned_df = df.dropna()
    X = cleaned_df.iloc[:, 1:25].applymap(lambda x: ord(x))
    Y = cleaned_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.7)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # print(clf.feature_importances_)

    # iris = load_iris()
    # model = KNeighborsClassifier(n_neighbors=3)
    # X_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    # model.fit(X_train, y_train)
    pickle.dump(clf, open("iris.pkl", "wb"))

    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    cap_shape = request.form['cap_shape']
    print(cap_shape)
    # sepal_width = 2  # request.form['sepal_width']
    # petal_length = 2  # request.form['petal_length']
    # petal_width = 2  # request.form['petal_width']
    # form_array = np.array(
    #     [[sepal_length, sepal_width, petal_length, petal_width]])
    model = pickle.load(open("iris.pkl", "rb"))
    me = pd.DataFrame([[cap_shape, 'b', 'a', 'b',
                        'b', 'a', 'b', 'e',
                        'e', 'b',  'b', 'a',
                        'a', 'b',  'b', 'a',
                        'a', 'b',  'b', 'a',
                        'b',  'b']]).applymap(lambda x: ord(x))
    prediction = model.predict(me)

    # if prediction == 0:
    #     result = "Iris Setosa"
    #     image = "iris-setosa.jpg"
    # elif prediction == 1:
    #     result = "Iris Versicolor"
    #     image = "iris-versicolor.jpg"
    # else:
    #     result = "Iris Virginica"
    #     image = "iris-virginica.jpg"

    return render_template("result.html", result=prediction, image="iris-versicolor.jpg")


if __name__ == "__main__":
    app.run(debug=True)


app = Flask(__name__)

#
# @app.route('/')
# def home():
#     iris = load_iris()
#     X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25)
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X_train, y_train)
#     pickle.dump(model,open('iris.pkl','wb'))
#     return render_template('home.html')
#
#
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     sepal_length = request.form['sepal_length']
#     sepal_width = request.form['sepal_width']
#     petal_length = request.form['petal_length']
#     petal_width = request.form['petal_width']
#
#     form_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     model = pickle.load(open('iris.pkl', 'rb'))
#     prediction = model.predict(form_array)[0]
#     if(prediction==0):
#         result="Iris Setosa"
#     elif(prediction==1):
#         result = "Iris Versicolour"
#     else:
#         result = "Iris Virginica"
#     return render_template('result.html', result=result)
#
#
# # @app.route('/predict', methods=['GET', 'POST'])
# # def predict():
# #     print("here")
# #     sepal_length = request.args['sepallength']
# #     print(sepal_length)
# #     sepal_width = request.form['sepal_width']
# #     petal_length = request.form['petal_length']
# #     petal_width = request.form['peal_width']
# #
# #     form_array = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
# #     model = pickle.load(open('iris.pkl','rb'))
# #     prediction = model.predict(form_array)
# #     return render_template('result.html',result=prediction)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
