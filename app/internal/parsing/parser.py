from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def get_links_from_google(query, browser):
    page = browser.new_page()
    page.goto(f'https://www.duckduckgo.com/{query}', timeout=30000)
    page.wait_for_load_state('networkidle')

    soup = BeautifulSoup(page.content(), 'html.parser')

    h2_headers = []
    links = []

    for h2 in soup.find_all('h2'):
        header_text = ''.join(str(content) for content in h2.contents if content.name != 'a')
        h2_headers.append(header_text.strip())


        link = h2.find('a')
        if link and 'href' in link.attrs:
            links.append(link['href'])

    return links[:7]


def get_text_from_url(url, browser):
    page = browser.new_page()
    page.goto(url, timeout=10000)

    page.wait_for_load_state('networkidle')

    soup = BeautifulSoup(page.content(), 'html.parser')
    text = soup.get_text()

    return text


def get_text(query) -> list:
    texts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, chromium_sandbox=True)
        urls = get_links_from_google(query, browser)

        for url in urls:
            try:
                texts.append(get_text_from_url(url, browser))
            except Exception as e:
                print(f"Ошибка при извлечении текста с {url}: {e}")

        browser.close()
    return texts


if __name__ == "__main__":
    query = "Куда сходить погулять в Москве"
    texts = get_text(query)

    # Выводим текст для каждого сайта
    for i, text in enumerate(texts):
        print(f"\nТекст с сайта {i + 1}:\n{text[:500]}...")
