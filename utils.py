import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from io import StringIO
import csv

def format_tanggal(bulan):
    now = datetime.now()
    return now - timedelta(days=bulan * 30)


def scrape_kompas(katakunci, jumlah):
    hasil_scraping = []
    for i in range(1, jumlah // 10 + 1):  # Ambil sesuai jumlah yang dimasukkan (10 berita per halaman)
        url = f"https://indeks.kompas.com/?site=all&q={katakunci}&page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('div', class_='article__list clearfix')
        for article in articles:
            title = article.find('h3', class_='article__title--medium').text.strip()
            link = article.find('a', class_='article__link')['href']
            date = article.find('div', class_='article__date').text.strip()

            # Scraping isi berita dari link
            content_response = requests.get(link)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='read__content')
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


def scrape_tribun(katakunci):
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
            title = article.find('h3').find('a')['title']

            # Scraping isi berita dari link
            content_response = requests.get(link)
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            content_paragraphs = content_soup.find_all('div', class_='side-article txt-article multi-fontsize')
            content = ' '.join([p.text.strip() for p in content_paragraphs])

            data = {
                'title': title,
                'date': date,
                'link': link,
                'content': content
            }

            hasil_scraping.append(data)
            print(data)  # Tambahkan pernyataan print di sini

    return hasil_scraping

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

def generate_csv(data):
    # Use the CSV module to create a CSV string from the data
    csv_string = StringIO()  # Use StringIO as a file-like object

    if data:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_string)

        # Write header
        header = ['Judul', 'Tanggal', 'Link', 'Isi Berita']
        csv_writer.writerow(header)

        # Write data rows
        for item in data:
            csv_writer.writerow([item['title'], item['date'], item['link'], item['content']])

    # Get the CSV content as a string
    csv_content = csv_string.getvalue()

    # Close the StringIO object
    csv_string.close()

    return csv_content