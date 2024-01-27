from flask import Flask, render_template, request, Response, url_for, redirect
from utils import  scrape_kompas, scrape_tempo, scrape_detik,scrape_tribun, generate_csv, generate_csv_processing
import pandas as pd
import os
from utils import preprocessing, Labelling, clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
global scraped_data
scraped_data = []
hasil_processing = []
vectorizer = TfidfVectorizer()
df_labeled = pd.DataFrame()

@app.route("/")
def index():
    return  render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit_form():
    source = request.form.get('source')
    kata_kunci = request.form.get('Kata Kunci')
    jumlah = int(request.form.get('Jumlah', 10))  # Nilai default adalah 10 jika tidak ada input

    if source == 'tribun':
        scraped_data  = scrape_tribun(kata_kunci, jumlah)
    elif source == 'tempo':
        scraped_data  = scrape_tempo(kata_kunci, jumlah)
    elif source == 'detik':
        scraped_data  = scrape_detik(kata_kunci, jumlah)
    elif source == 'kompas':
        scraped_data  = scrape_kompas(kata_kunci, jumlah)

    return render_template('scraping.html', data=scraped_data )

@app.route("/scraping")
def route_scraping():
    return render_template('scraping.html', data=scraped_data)


@app.route('/download', methods=['POST'])
def download():
    csv_data = generate_csv(scraped_data)
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=news_data.csv'}
    )
    return response


@app.route("/processing")
def processing():
    return render_template('processingData.html', data=hasil_processing)

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        df = pd.read_csv(request.files['file'])
        df['text_tokenize'] = df['content'].apply(preprocessing)
        df['text_clean'] = df['content'].apply(clean_text)

        # Melakukan tahap Labelling
        lexicon_df = pd.read_csv('static/lexicon-word-dataset.csv')  # Ganti dengan path yang sesuai
        labeller = Labelling(df.to_dict(orient='records'), lexicon_df)
        df_labeled = labeller.labelling_data()

        # Simpan hasil processing di dalam variabel global hasil_processing
        global hasil_processing
        hasil_processing = df_labeled.to_dict(orient='records')

        # Menampilkan hasil ke halaman
        return render_template('processingData.html', data=hasil_processing)

@app.route('/download_processing', methods=['POST'])
def download_processing():
    # Generate CSV data
    csv_data = generate_csv_processing(hasil_processing)

    # Create a response with the CSV data
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=Hasil_Processing.csv'}
    )

    return response

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    global hasil_processing, vectorizer, knn_model

    if request.method == 'POST':
        uploaded_file = request.files['file']
        n_neighbors = int(request.form['n_neighbors'])
        split_ratio = float(request.form['split_ratio'])

        if uploaded_file.filename != '':
            df = pd.read_csv(uploaded_file)
            df['text_tokenize'] = df['content'].apply(preprocessing)
            df['text_clean'] = df['content'].apply(clean_text)
            lexicon_df = pd.read_csv('static/lexicon-word-dataset.csv')
            labeller = Labelling(df.to_dict(orient='records'), lexicon_df)
            df_labeled = labeller.labelling_data()
            X = vectorizer.fit_transform(df_labeled['text_clean'])
            y = df_labeled['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_model.fit(X_train, y_train)
            hasil_processing = df_labeled.to_dict(orient='records')
            return redirect(url_for('hasil_sentimen'))
    return render_template('knn.html')

@app.route('/hasil_sentimen')
def hasil_sentimen():
    global hasil_processing, vectorizer, knn_model, df_labeled

    if df_labeled.empty:
        df_labeled = pd.DataFrame(hasil_processing)
        X = vectorizer.fit_transform(df_labeled['text_clean'])
        y = df_labeled['label']
        knn_model.fit(X, y)
    try:
        text_vectorized = vectorizer.transform(["dummy text"])
    except AttributeError:
        df_labeled = pd.DataFrame(hasil_processing)
        X = vectorizer.fit_transform(df_labeled['text_clean'])
        y = df_labeled['label']
        knn_model.fit(X, y)

    hasil_sentimen = []
    actual_labels = []
    for data in hasil_processing:
        text_vectorized = vectorizer.transform([data['text_clean']])
        prediction = knn_model.predict(text_vectorized)[0]
        hasil_sentimen.append({'title': data['title'], 'sentiment': prediction})
        actual_labels.append(data['label'])

    cm = confusion_matrix(actual_labels, [result['sentiment'] for result in hasil_sentimen])
    accuracy = accuracy_score(actual_labels, [result['sentiment'] for result in hasil_sentimen])
    precision = precision_score(actual_labels, [result['sentiment'] for result in hasil_sentimen], average='weighted')
    recall = recall_score(actual_labels, [result['sentiment'] for result in hasil_sentimen], average='weighted')
    f1 = f1_score(actual_labels, [result['sentiment'] for result in hasil_sentimen], average='weighted')

    # Create Word Cloud
    all_text = ' '.join(df_labeled['text_clean'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Ensure the 'static/image' directory exists
    image_dir = 'static/image'
    os.makedirs(image_dir, exist_ok=True)

    # Save the Word Cloud image
    wordcloud_path = os.path.join(image_dir, 'wordcloud.png')
    plt.savefig(wordcloud_path)
    plt.close()

    # Create Pie Chart
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [cm[0, 0], cm[1, 1], cm[2, 2]]
    explode = (0.1, 0, 0)  # explode the 1st slice (Positive)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Ensure the 'static/image' directory exists
    image_dir = 'static/image'
    os.makedirs(image_dir, exist_ok=True)

    # Save the Pie Chart image
    pie_chart_path = os.path.join(image_dir, 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()

    return render_template('knn.html', hasil_sentimen=hasil_sentimen, confusion_matrix=cm,
                           accuracy=accuracy, precision=precision, recall=recall, f1=f1, wordcloud_path=wordcloud_path, pie_chart_path=pie_chart_path)

@app.route('/svm')
def svm():
    return render_template('svm.html')


@app.route('/hasil')
def hasil():
    return render_template('hasil.html')

if __name__ == "__main__":
    app.run(debug=True)
