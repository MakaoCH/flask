from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import requests
import json
from config import BEARER_TOKEN
from werkzeug.utils import secure_filename
import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from collections import defaultdict
import keras
import cv2
import numpy as np



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOADS'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prenom = db.Column(db.String(50), nullable=False)
    nom = db.Column(db.String(50), nullable=False)
    sexe = db.Column(db.String(10), nullable=False)
    pseudo = db.Column(db.String(50), nullable=False)


with app.app_context():
    db.create_all()

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/reponse_formulaire', methods=['POST'])
def rep_form():
    sexe = request.form['sexe']
    prenom = request.form['prenom']
    nom = request.form['nom']
    pseudo = request.form['pseudo']
           
    existing_user = User.query.filter_by(pseudo=pseudo).first()

    if existing_user:
        return render_template("pseudo_exists.html", pseudo=pseudo)

    if sexe == "Homme":
        titre = "Mr"
    else:
        titre = "Mme"

    message = f"{titre} {prenom} {nom}, votre nom d'utilisateur est {pseudo}"
    
    user = User(prenom=prenom, nom=nom, sexe=sexe, pseudo=pseudo)
    db.session.add(user)
    db.session.commit()

    return render_template("page2.html", message=message)


@app.route('/inscrits', methods=['GET'])
def users():
    all_users = User.query.all()
    return render_template("inscrits.html", users=all_users)

@app.route("/form_actualite")
def form_actu():
    return render_template("form_actualite.html")

@app.route('/actualites', methods=['POST'])
def get_news():
    entreprise_symbole = request.form.get('entreprise_symbole')

    if not entreprise_symbole:
        return render_template("erreur.html", message="Veuillez entrer le symbole de l'entreprise.")

    url = 'https://devapi.ai/api/v1/markets/news'
    params = {
        'ticker': entreprise_symbole,
    }
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        news_data = response.json()
        
        articles = news_data.get('body', [])

        return render_template("actualites.html", entreprise_symbole=entreprise_symbole, articles=articles)
    else:
        return render_template("erreur.html", message=f"Erreur lors de la récupération des actualités. Code d'état : {response.status_code}")

@app.route("/form_bourse")
def form_bourse():
    return render_template("form_bourse.html") 

@app.route('/bourses', methods=['POST'])
def get_bourse():
    entreprise_symbole = request.form.get('entreprise_symbole') 

    if not entreprise_symbole:
        return render_template("erreur.html", message="Veuillez entrer le symbole de l'entreprise.")

    url = 'https://devapi.ai/api/v1/markets/quote'
    params = {
        'ticker': entreprise_symbole,
    }
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        news_data = response.json()
        
        quotes = news_data.get('body', [])
        primary_data = news_data.get('primaryData', {}) 
 
        return render_template("bourses.html", entreprise_symbole=entreprise_symbole, quotes=quotes, primary_data=primary_data)
    else:
        return render_template("erreur.html", message=f"Erreur lors de la récupération des informations sur la bourse. Code d'état : {response.status_code}")


def stats(file_path):
    try:
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') or file_path.endswith('.xls') else pd.read_csv(file_path)
        statistics = df.describe()
        return statistics.to_html()
    except Exception as e:
        return f"Erreur: {str(e)}"
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        statistics = stats(file_path)

        # Remove uploaded file after getting statistics
        os.remove(file_path)

        return render_template('download.html', statistics=statistics)
    else:
        return 'Format de fichier invalide. Veuillez télécharger un fichier CSV ou Excel.'

@app.route('/stocks')
def stocks():
    return render_template('stocks.html')

from datetime import datetime

@app.route('/history', methods=['POST'])
def history():
    entreprise_symbole = request.form.get('entreprise_symbole')
    interval = request.form.get('interval') 

    if not entreprise_symbole:
        return render_template("erreur.html", message="Veuillez entrer le symbole de l'entreprise et la période.")

    url = 'https://devapi.ai/api/v1/markets/stock/history'
    params = {
        'ticker': entreprise_symbole,
        'interval': interval,
    }
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }
    response = requests.request('GET', url, headers=headers, params=params)

    if response.status_code == 200:
        news_data = response.json()
        
        prices = news_data.get('body', [])

        aggregated_prices = defaultdict(list)
        for entry in prices.values():
            date_str = entry.get('date')
            close = entry.get('close')
            
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")

            # Extraire le mois de la date
            month_key = date_obj.strftime('%Y-%m')
            
            aggregated_prices[month_key].append(close)

        month_labels = list(aggregated_prices.keys())
        close_values = [sum(prices) / len(prices) for prices in aggregated_prices.values()]
 
        plt.figure(figsize=(12, 7))
        plt.plot(month_labels, close_values, label='Close Price')
        plt.ylabel('Close Price')
        plt.xticks(rotation=90)
        plt.legend()
        
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()

        base64_img = base64.b64encode(img_data.read()).decode('utf-8')

        return render_template("stocks_graph.html", entreprise_symbole=entreprise_symbole, base64_img=base64_img)
    else:
        return render_template("erreur.html", message=f"Erreur lors de la récupération des informations sur la bourse. Code d'état : {response.status_code}")

cnn = keras.models.load_model('mnist_model.h5')

def process(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (32, 32))
    image = np.resize(image, (1, 32, 32, 3))
    image = image/255.0
    image = 1-image
    return image

@app.route('/pict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOADS'], file.filename)

        # Créez le dossier 'uploads' s'il n'existe pas
        if not os.path.exists(app.config['UPLOADS']):
            os.makedirs(app.config['UPLOADS'])

        file.save(filepath)
        image = process(filepath)
        print('process done')
        prediction = np.argmax(cnn.predict(image), axis=-1)

        return render_template('mnist.html', number=prediction[0])

    return render_template('mnist.html')

@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if request.method == 'POST':
        # Récupérer l'image du dessin depuis la requête
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOADS'], 'drawn_digit.png')

        # Créez le dossier 'uploads' s'il n'existe pas
        if not os.path.exists(app.config['UPLOADS']):
            os.makedirs(app.config['UPLOADS'])

        file.save(filepath)

        # Prétraitement de l'image si nécessaire
        drawn_image = process(filepath)

        # Effectuer la prédiction à l'aide du modèle
        prediction = np.argmax(cnn.predict(drawn_image), axis=-1)

        return jsonify({'prediction': int(prediction)})

    return render_template('draw.html')

    
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
