#!/bin/env python3
from collections import defaultdict
import sys
from time import sleep
from typing import Dict, List, Tuple, Any
import requests
import re
from readability import readability  # type: ignore
import os
from multiprocessing import Pool, Lock, Manager
from keybert import KeyBERT
import json
from bs4 import BeautifulSoup
import logging
from fuzzywuzzy import fuzz

Document = readability.Document
ManagerT = Any

Keyword = Tuple[str, float]
Keywords = List[Keyword]


def load_content_cache(manager: ManagerT) -> Dict[str, str]:
    if not os.path.exists('content_cache.txt'):
        return {}

    with open('content_cache.txt', 'r') as file:
        content = file.read()
        d = manager.dict()
        d.update(json.loads(content))
        return d


def update_content_cache(cache: Dict[str, str]) -> None:
    update_content_cache.lock.acquire()

    with open('content_cache.txt', 'w') as file:
        file.write(json.dumps(cache, indent=4, sort_keys=True))

    update_content_cache.lock.release()


update_content_cache.lock = Lock()


def load_keywords_cache(manager: ManagerT) -> Dict[str, Keywords]:
    if not os.path.exists('keywords_cache.txt'):
        return {}

    with open('keywords_cache.txt', 'r') as file:
        content = file.read()
        d = manager.dict()
        d.update(json.loads(content))
        return d


def update_keywords_cache(cache: Dict[str, Keywords]) -> None:
    update_keywords_cache.lock.acquire()

    with open('keywords_cache.txt', 'w') as file:
        file.write(json.dumps(cache, indent=4, sort_keys=True))

    update_keywords_cache.lock.release()


update_keywords_cache.lock = Lock()


def get_content(link: str) -> str:
    header: Dict[str, str] = {
        'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    response = requests.get(link, headers=header)
    response.raise_for_status()

    doc = Document(response.content)
    html = doc.summary()  # type: ignore

    soup = BeautifulSoup(html, features='html.parser')  # type: ignore
    text = soup.get_text()
    if len(text) < 500:  # small text, probably not a good article
        return doc.title()
    return text


def parse_markdown_links(file_path: str) -> Dict[str, str]:
    links: Dict[str, str] = {}
    with open(file_path, 'r') as file:
        content = file.read()
        # Regular expression to match markdown links
        pattern = r'\[(.*)\]\((.*)\)'
        matches = re.finditer(pattern, content)
        for match in matches:
            text = match.group(1)
            link = match.group(2)
            links[text] = link
    return links


def get_keywords(content: str) -> Keywords:
    kw_model = KeyBERT()  # type: ignore
    return kw_model.extract_keywords(  # type: ignore
        content, top_n=5)


def process_link(link: str, contents: Dict[str, str],
                 keywords: Dict[str, Keywords]) -> None:
    if link not in contents:
        contents[link] = get_content(link)

    if link not in keywords:
        keywords[link] = get_keywords(contents[link])


def process_links(parsed_links: Dict[str, str]) -> Dict[str, Keywords]:
    with Pool(12) as pool, Manager() as manager:
        contents = load_content_cache(manager)
        keywords = load_keywords_cache(manager)

        def update_caches() -> None:
            update_content_cache(dict(contents))
            update_keywords_cache(dict(keywords))

        results: List[Any] = []
        for link in parsed_links.values():
            res = pool.apply_async(process_link,
                                   args=(link, contents, keywords))
            results.append((res, link))

        pool.close()

        while len(results) > 0:
            sleep(10)

            print(
                f"Waiting for content to be parsed. Content: {len(contents)}/{len(parsed_links)} ; Keywords: {len(keywords)}/{len(parsed_links)}"
            )

            for res, link in results:
                if res.ready() and not res.successful():
                    try:
                        res.get()
                    except Exception as e:
                        logging.exception(e)
                        contents[link] = ''
                        keywords[link] = [('ERROR', 0.0)]

            results = [(res, link) for res, link in results if not res.ready()]

            update_caches()

        pool.join()
        update_caches()

        return dict(keywords)


def simplify_keywords(keywords: Dict[str, Keywords]) -> Dict[str, List[str]]:
    THRESHOLD = 88  # threshold for similarity
    for link1, kwds in keywords.items():
        for i, (kw1, str1) in enumerate(kwds):
            for link2, kwds2 in keywords.items():
                for j, (kw2, str2) in enumerate(kwds2):
                    if link1 == link2 and i == j:
                        continue
                    ratio: int = fuzz.ratio(kw1, kw2)  # type: ignore
                    if ratio > THRESHOLD:
                        # keep the shorter keyword
                        if len(kw1) > len(kw2):
                            kw1, kw2 = kw2, kw1

                        keywords[link1][i] = (kw1, str1)
                        keywords[link2][j] = (kw1, str2)

    keywords_no_strength: Dict[str, List[str]] = {}
    # remove strength and duplicates
    for link, kwds in keywords.items():
        keywords_no_strength[link] = list(set([x[0] for x in kwds]))

    return keywords_no_strength


def group_links_by_domain(links: Dict[str, str]) -> Dict[str, List[str]]:
    result_domain: defaultdict[str, List[str]] = defaultdict(list)
    for link in links.values():
        domain = re.match(r'https?://([^/]+)', link).group(1)  # type: ignore
        result_domain[domain].append(link)

    return result_domain


def print_to_md(title_to_link: Dict[str, str], domain_to_link: Dict[str,
                                                                    List[str]],
                keywords: Dict[str, List[str]]) -> None:

    link_to_title = {v: k for k, v in title_to_link.items()}

    with open('result.md', 'w') as file:
        for domain, links in domain_to_link.items():
            file.write(f'# {domain} ({len(links)})\n')
            for link in links:
                link_title = link_to_title[link]
                keywords_str = ', '.join(keywords[link])
                file.write(f'- [{link_title}]({link}) - {keywords_str}\n')
            file.write('\n')


def main() -> None:
    file_path = sys.argv[1]
    parsed_links = parse_markdown_links(file_path)

    keywords = process_links(parsed_links)
    keywords = simplify_keywords(keywords)

    with open('keywords.json', 'w') as file:
        json.dump(keywords, file, indent=4, sort_keys=True)

    domains = group_links_by_domain(parsed_links)
    print_to_md(parsed_links, domains, keywords)


if __name__ == '__main__':
    main()