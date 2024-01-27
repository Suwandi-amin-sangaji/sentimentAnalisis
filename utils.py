import string
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from io import StringIO
import csv
import re
import pandas as pd

from nltk.corpus import stopwords
import re
import string
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def format_tanggal(bulan):
    now = datetime.now()
    return now - timedelta(days=bulan * 30)


def scrape_kompas(katakunci, jumlah):
    hasil_scraping = []
    for i in range(1, jumlah // 10 + 1):  # Ambil sesuai jumlah yang dimasukkan (10 berita per halaman)
        url = f"https://search.kompas.com/search/?q={katakunci}#gsc.tab=0&gsc.q={katakunci}&gsc.page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('div', class_='gsc-webResult gsc-result')
        for article in articles:
            title = article.find('div', class_='gs-title').text.strip()
            link = article.find('a')['href']
            date = article.find('div', class_='gs-bidi-start-align gs-visibleUrl gs-visibleUrl-breadcrumb').text.strip()

            # Scraping isi berita dari link
            content_response = requests.get(link)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='gsc-table-result')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            if len(hasil_scraping) >= jumlah:
                break

    return hasil_scraping[:jumlah] 

def scrape_tribun(katakunci, jumlah):
    hasil_scraping = []
    for i in range(1, jumlah // 10 + 1):  # Ambil sesuai jumlah yang dimasukkan (10 berita per halaman)
        url = f"https://www.tribunnews.com/search?q={katakunci}&page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('div', class_='gsc-resultsRoot gsc-tabData gsc-tabdActive')
        for article in articles:
            link = article.find('h3').find('a')['href']
            title = article.find('h3').find('a')['gs-title']
            date = article.find('div', class_='gs-bidi-start-align gs-snippet').text.strip()

           # Scraping isi berita dari link
            content_response = requests.get(link)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='gsc-table-result')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            if len(hasil_scraping) >= jumlah:
                break

    return hasil_scraping[:jumlah] 

    hasil_scraping = []
    for i in range(1, 3):  # Ambil halaman 1 dan 2
        url = f"https://www.tribunnews.com/search?q={katakunci}&page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('li', class_='ptb15')
        for article in articles:
            time = article.find('time', class_='grey').text.split()
            if time[2].lower() == 'oktober':
                time[2] = 10
            elif time[2].lower() == 'november':
                time[2] = 11
            elif time[2].lower() == 'desember':
                time[2] = 12

            date = f"{time[1]}/{time[2]}/{time[3]} {time[4]} {time[5]}"
            link = article.find('h3').find('a')['href']
            title = article.find('h3').find('a')['gs-title']

            # Scraping isi berita dari link
            content_response = requests.get(link)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='gsc-table-result')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            if len(hasil_scraping) >= jumlah:
                break # Tambahkan pernyataan print di sini

    return hasil_scraping[:jumlah]

def scrape_detik(katakunci, jumlah):
    hasil_scraping = []
    latest_date = None  # Variable to store the latest date

    for page in range(1, jumlah // 10 + 1):
        url = f'https://www.detik.com/search/searchall?query={katakunci}&siteid=2&page={page}'
        
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('article')
        for article in articles:
            link = article.find('a')['href']
            date = article.find('span', class_='date').text.replace('WIB', '').strip()
            title = article.find('h2').text.strip()

            # Scraping isi berita dari link
            content_response = requests.get(link, verify=False)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='detail__body-text itp_bodycontent')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            if len(hasil_scraping) >= jumlah:
                break

    return hasil_scraping[:jumlah]

def scrape_tempo(katakunci, jumlah):
    hasil_scraping = []

    for page in range(1, jumlah // 10 + 1):
        url = f'https://www.tempo.co/search?q={katakunci}&page={page}'

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('article', class_='text-card')
        for article in articles:
            link = article.find('a')['href']

            title_container = article.find('h2', class_='title')
            title = title_container.text.replace('\n', '').strip() if title_container else 'No title found'


            date_container = article.find('h4', class_="date")
            date = date_container.text.strip() if date_container else ''

             # Scraping isi berita dari link
            content_response = requests.get(link, verify=False)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='detail-in')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            if len(hasil_scraping) >= jumlah:
                break

    return hasil_scraping[:jumlah]

def generate_csv(data):
    csv_string = StringIO()
    if data:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_string)

        # Write header
        header = ['title', 'date', 'link', 'content']
        csv_writer.writerow(header)

        # Write data rows
        for item in data:
            csv_writer.writerow([item['title'], item['date'], item['link'], item['content']])

    csv_content = csv_string.getvalue()
    csv_string.close()

    return csv_content


def generate_csv_processing(data):
    csv_string = StringIO()  # Use StringIO as a file-like object

    if data:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_string)

        header = ['title', 'date', 'link', 'content', 'text_tokenize', 'text_clean', 'label', 'score']
        csv_writer.writerow(header)

        # Write data rows
        for item in data:
            csv_writer.writerow([item['title'], item['date'], item['link'], item['content'], item['text_tokenize'], item['text_clean'], item['label'], item['score']])

    csv_content_processing = csv_string.getvalue()
    # Close the StringIO object
    csv_string.close()

    return csv_content_processing

# Fungsi untuk tahap Cleansing
def preprocessing(text):
    # 1. Cleansing
    text = clean_text(text)

    # 2. Stopword Removal, Case Folding, Tokenizing
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # 3. Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words

def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', ' ', text)
        return text 
    else:
        return str(text)



class Labelling:
    def __init__(self, dataset, lexicon_df):
        self.dataset = dataset
        self.lexicon_df = lexicon_df

    def labelling_data(self):
        df = pd.DataFrame(self.dataset)
        df['label'], df['score'] = zip(*[self.label_lexicon(df['text_clean'][i]) for i in range(len(df))])
        return df

    def label_lexicon(self, text):
        words = text.split()
        labels = []

        for word in words:
            label = self.lexicon_df.loc[self.lexicon_df['word'] == word, 'weight'].values
            if len(label) > 0:
                labels.append(label[0])

        sentiment_score = sum(labels)
        if sentiment_score > 0:
            return 'Positive', sentiment_score
        elif sentiment_score < 0:
            return 'Negative', sentiment_score
        else:
            return 'Netral', sentiment_score

