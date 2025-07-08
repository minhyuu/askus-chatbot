import requests
import time
from bs4 import BeautifulSoup
import random

URL = "https://askus.utas.edu.au"

    
def extract_title(soup):
    """
    Extracts the title from the BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page.

    Returns:
        str: The title of the page.
    """
    title = soup.find('h1').get_text(strip=False) if soup.find('h1') else "No title found"
    return title


def extract_content(soup):
    """
    Extracts the main content from the BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page.

    Returns:
        str: The main content of the page.
    """
    content = soup.find('div', class_='richtext richtext__large')
    #remove spaces in the end
    if content:
        content = content.get_text(strip=False).replace('\n', ' ').strip()
    else:
        content = None
    return content


def crawl_faq_page(url):
    """
    Crawls the FAQ page and data in json-like format.

    Args:
        url (str): The URL of the FAQ page.

    Returns:
        tuple: data in json-like format including title, content, source, token_count.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # extract title
        title = extract_title(soup)
        # print(f"Title: {title}")

        # extract content
        content = extract_content(soup)
        # print(f"Content: {content}")

        # count tokens in content
        token_count = len(content.split())

        data = {
            'title': title,
            'content': content,
            'source': url,
            'token_count': token_count
        }
        return data

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def crawl_content(url):
    """
    Crawls the content of a given URL and returns the text content.

    Args:
        url (str): The URL to crawl.

    Returns:
        str: The text content of the page.
    """
    try:
        faq_content = [] # List to store FAQ content

        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # extract h3
        h3_elements = soup.find_all('h3')
        for h3 in h3_elements:
            link = h3.find('a')['href'] if h3.find('a') else None
            parsed_link = URL + link if link else "No link found"

            faq_data = crawl_faq_page(parsed_link)
            faq_content.append(faq_data)

            seconds = random.randint(3, 10)  # Random sleep time between 3 and 30 seconds
            time.sleep(seconds)  # Sleep to avoid overwhelming the server
            print(f"Processed: {parsed_link} \n Sleep for {seconds} seconds")

        return faq_content
        
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    

if __name__ == "__main__":
    knowledge_base = [] # List to store all knowledge base entries

    print("Starting to crawl the FAQ pages...")


    # # enrollment
    # for page_number in range(1, 10):
    #     faq_link = f"https://askus.utas.edu.au/app/answers/list/st/5/kw/enrol/page/{page_number}"
    #     data = crawl_content(faq_link) 
    #     knowledge_base.extend(data) # Add the crawled data to the knowledge base

    # # timetable
    # for page_number in range (1, 4): 
    #     faq_link = f"https://askus.utas.edu.au/app/answers/list/st/5/kw/timetable/page/{page_number}"
    #     data = crawl_content(faq_link) 
    #     knowledge_base.extend(data)  # Add the crawled data to the knowledge base

    # # login & password:
    # for page_number in range (1, 4): 
    #     faq_link = f"https://askus.utas.edu.au/app/answers/list/st/5/kw/password%20login/page/{page_number}"
    #     data = crawl_content(faq_link) 
    #     knowledge_base.extend(data)  # Add the crawled data to the knowledge base

    # # fees:
    # for page_number in range (1, 9): 
    #     faq_link = f"https://askus.utas.edu.au/app/answers/list/st/5/kw/fees/page/{page_number}"
    #     data = crawl_content(faq_link) 
    #     knowledge_base.extend(data)  # Add the crawled data to the knowledge base

    faq_link = f"https://askus.utas.edu.au/app/answers/list/st/5/kw/fees/page/8"
    data = crawl_content(faq_link) 
    knowledge_base.extend(data)  # Add the crawled data to the knowledge base

    # convert knowledge base to json
    import json
    with open('knowledge_base_4.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    print("Knowledge base has been saved to 'knowledge_base.json'.")
    print(f"Total entries in knowledge base: {len(knowledge_base)}")
    print("Crawling completed.")