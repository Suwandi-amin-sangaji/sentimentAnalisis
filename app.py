from flask import Flask, render_template, request, Response
from utils import  scrape_kompas, scrape_tribun, scrape_detik, generate_csv

app = Flask(__name__)
scraped_data = []

@app.route("/")
def index():
    return  render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit_form():
    global scraped_data
    source = request.form.get('source')
    kata_kunci = request.form.get('Kata Kunci')
    jumlah = int(request.form.get('Jumlah', 10))  # Nilai default adalah 10 jika tidak ada input

    if source == 'kompas':
        scraped_data  = scrape_kompas(kata_kunci, jumlah)
    elif source == 'tribun':
        scraped_data  = scrape_tribun(kata_kunci, jumlah)
    elif source == 'detik':
        scraped_data  = scrape_detik(kata_kunci, jumlah)

    return render_template('scraping.html', data=scraped_data )

@app.route("/scraping")
def route_scraping():
    return render_template('scraping.html', data=scraped_data )

@app.route('/download', methods=['POST'])
def download():
    # Generate CSV data
    csv_data = generate_csv(scraped_data)

    # Create a response with the CSV data
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=news_data.csv'}
    )

    return response

@app.route("/processing")
def processing():
    return render_template('processingData.html')

@app.route('/knn')
def knn():
    # Senntiment analisis dengan menggunakan Knn
    return render_template('knn.html')

@app.route('/svm')
def svm():
    return render_template('svm.html')


@app.route('/hasil')
def hasil():
    return render_template('hasil.html')

if __name__ == "__main__":
    app.run(debug=True)
