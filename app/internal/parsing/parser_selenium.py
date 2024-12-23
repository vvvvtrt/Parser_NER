from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def get_links_from_google(query, driver):
    driver.get(f'https://www.duckduckgo.com/?q={query}')

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    h2_headers = []
    links = []

    for h2 in soup.find_all('h2'):
        header_text = ''.join(str(content) for content in h2.contents if content.name != 'a')
        h2_headers.append(header_text.strip())

        link = h2.find('a')
        if link and 'href' in link.attrs:
            links.append(link['href'])

    return links[:7]

def get_text_from_url(url, driver):
    driver.get(url)
    WebDriverWait(driver, 10).until(lambda x: True)  # Ждем 10 секунд

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'html.parser')
    text = soup.get_text()

    return text

def get_text(query) -> list:
    texts = []

    options = ChromeOptions()
    options.page_load_strategy = 'eager'

    # options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    urls = get_links_from_google(query, driver)

    for url in urls:
        try:
            texts.append(get_text_from_url(url, driver))
        except Exception as e:
            print(f"Ошибка при извлечении текста с {url}: {e}")

    driver.quit()
    return texts

if __name__ == "__main__":
    query = "Куда сходить погулять в Москве"
    texts = get_text(query)

    # Выводим текст для каждого сайта
    for i, text in enumerate(texts):
        print(f"\nТекст с сайта {i + 1}:\n{text[:500]}...")
