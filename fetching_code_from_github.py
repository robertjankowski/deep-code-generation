import configparser
import requests

from github import Github
from typing import List


class ConfigNotFoundException(Exception):
    pass


def get_github_access_token(filename: str):
    config = configparser.ConfigParser()
    config.read(filename)
    try:
        return config.get('github', 'access_token')
    except:
        raise ConfigNotFoundException(
            'Field [access_token] in section [github] not found')


def get_repos_urls_by_query(g: Github, query: List[str]):
    query = '+'.join(query) + '+in:readme+in:description'
    result = g.search_repositories(query, 'stars', 'desc')

    print(f'Found {result.totalCount} repos')
    return [repo.clone_url for repo in result]


def get_files_by_extension(g: Github, query: List[str], extension: str, max_size=100):
    rate_limit = g.get_rate_limit()
    rate = rate_limit.search
    print(f'{rate.remaining}/{rate.limit} API calls remaining')

    query = '+'.join(query) + '+in:file extension:' + extension
    result = g.search_code(query, order='desc')

    print(f'Found {result.totalCount} files')
    if result.totalCount > max_size:
        result = result[:max_size]

    return [file.download_url for file in result]


def download_code(path: str, links: List[str]):
    for link in links:
        r = requests.get(link)
        open(path + '/' + link.split('/')[-1], 'wb').write(r.content)


if __name__ == "__main__":
    ACCESS_TOKEN = get_github_access_token('github_credentials.ini')

    g = Github(ACCESS_TOKEN)
    scala_files = get_files_by_extension(g, ['deep learning'], 'py', max_size=100)

    github_source_path = 'data/github_sources'
    download_code(github_source_path, scala_files)
