from flask import Flask, render_template, request
import re,nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import pickle

cv = pickle.load(open("vectorize.pkl", "rb"))
clf = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)


@app.route('/')
def index():
    # Vectorize the input
    # result = cv.transform([sample]).toarray()
    # Predict
    # pred = clf.predict(result)
    # print(pred)
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    userInput = request.form.get('email')
    desmos = []
    rev = re.sub('[^a-zA-Z]', ' ', userInput)
    rev = rev.lower()
    rev = rev.split(" ")

    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    desmos.append(rev)
    test = cv.transform(desmos).toarray()
    pred = clf.predict(test)
    pred = pred[0]
    # Predict
    if pred == 0:
        pred = -1
    return render_template("second.html", label=pred)


if __name__ == "__main__":
    app.run(debug=True)