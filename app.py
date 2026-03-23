from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("gwp.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/pred", methods=["POST"])
def pred():
    quarter = request.form["quarter"]
    department = request.form["department"]
    day = request.form["day"]
    team = request.form["team"]
    targeted_productivity = request.form["targeted_productivity"]
    smv = request.form["smv"]
    over_time = request.form["over_time"]
    incentive = request.form["incentive"]
    idle_time = request.form["idle_time"]
    idle_men = request.form["idle_men"]
    no_of_style_change = request.form["no_of_style_change"]
    no_of_workers = request.form["no_of_workers"]
    month = request.form["month"]

    total = [[
        int(quarter),
        int(department),
        int(day),
        int(team),
        float(targeted_productivity),
        float(smv),
        int(over_time),
        int(incentive),
        float(idle_time),
        int(idle_men),
        int(no_of_style_change),
        float(no_of_workers),
        int(month)
    ]]

    prediction = model.predict(total)[0]

    if prediction <= 0.3:
        text = "low"
    elif prediction <= 0.8:
        text = "medium"
    else:
        text = "high"

    return render_template("submit.html", prediction=text)

if __name__ == "__main__":
    app.run(debug=True)


