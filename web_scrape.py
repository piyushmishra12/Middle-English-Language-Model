from bs4 import BeautifulSoup
import requests

link = "https://archive.org/stream/canterburytaleso00chauuoft/canterburytaleso00chauuoft_djvu.txt"
canterbury_tales = requests.get(link)

soup = BeautifulSoup(canterbury_tales.text, 'html.parser')
# print(soup.pre.prettify())

with open('text.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(soup.pre.prettify())