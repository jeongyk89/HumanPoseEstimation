import requests
from bs4 import BeautifulSoup

url = "https://www.kth.se/en/aktuellt/nyheter"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

titles = soup.find_all("h3", class_="title")

for title in titles:
    print(title.text.strip())
