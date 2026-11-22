"""Cache exact official CUDA wheels with restartable downloads and SHA256 checks."""
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
import hashlib
import importlib.metadata as installed_metadata
import html
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from packaging.requirements import Requirement

root = Path(__file__).resolve().parent
cache = root / '.wheels'
cache.mkdir(exist_ok=True)


def fetch(url):
    for attempt in range(8):
        try:
            with urlopen(url, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 7:
                raise
            time.sleep(3)


def official_torch_wheel(package):
    index = f'https://download.pytorch.org/whl/cu124/{package}/'
    page = fetch(index).decode()
    filename = f'{package}-2.6.0+cu124-cp310-cp310-linux_x86_64.whl'
    href = next(html.unescape(link) for link in re.findall(r'href="([^"]+)"', page)
                if filename in unquote(link))
    url, fragment = urldefrag(urljoin(index, href))
    digest = fragment.removeprefix('sha256=')
    assert re.fullmatch('[a-f0-9]{64}', digest), href
    return {'filename': filename, 'url': url, 'sha256': digest}


torch = official_torch_wheel('torch')
audio = official_torch_wheel('torchaudio')
metadata = Parser().parsestr(fetch(torch['url'] + '.metadata').decode())
dependencies = []
for requirement in metadata.get_all('Requires-Dist', []):
    parsed = Requirement(requirement)
    if parsed.name.startswith('nvidia-') or parsed.name == 'triton':
        pins = list(parsed.specifier)
        assert len(pins) == 1 and pins[0].operator == '==', requirement
        dependencies.append((parsed.name, pins[0].version))
assert dependencies, 'CUDA dependencies absent from official torch metadata'


def pypi_wheel(spec):
    name, version = spec
    info = json.loads(fetch(f'https://pypi.org/pypi/{name}/{version}/json'))
    candidates = [item for item in info['urls'] if item['filename'].endswith('.whl')
                  and 'linux' in item['filename'] and 'x86_64' in item['filename']
                  and ('py3-none' in item['filename'] or 'cp310-cp310' in item['filename'])]
    assert candidates, (name, version)
    item = sorted(candidates, key=lambda x: x['filename'])[0]
    return {'filename': item['filename'], 'url': item['url'],
            'sha256': item['digests']['sha256'], 'size': item['size']}


with ThreadPoolExecutor(max_workers=4) as pool:
    manifest = list(pool.map(pypi_wheel, dependencies)) + [torch, audio]
(root / 'results/env_download_manifest.json').write_text(json.dumps(manifest, indent=2))


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(item):
    target = cache / item['filename']
    if not target.exists():
        name, version = item['filename'].split('-', 2)[:2]
        try:
            distribution = installed_metadata.distribution(name)
        except installed_metadata.PackageNotFoundError:
            distribution = None
        if distribution is not None and distribution.version == version:
            files = distribution.files
            if (files and any(str(f).endswith('.dist-info/RECORD') for f in files)
                    and all(distribution.locate_file(f).exists() for f in files)):
                print('Already installed completely:', name, version, flush=True)
                return
    if target.exists():
        assert checksum(target) == item['sha256'], f'Cached hash mismatch: {target}'
        print('Verified cached:', target.name, flush=True)
        return
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(50):
        offset = partial.stat().st_size if partial.exists() else 0
        print(f'Downloading {target.name}: resume {offset / 1e6:.1f} MB, attempt {attempt + 1}', flush=True)
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with urlopen(Request(item['url'], headers=headers), timeout=90) as response:
                if offset:
                    assert response.status == 206, 'Server did not honor resume range'
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                expected = offset + int(response.headers['Content-Length'])
                if 'size' in item:
                    assert expected == item['size']
                with partial.open('ab') as file:
                    report_at = offset + 32 * 1024 ** 2
                    while True:
                        chunk = response.read(1024 ** 2)
                        if not chunk:
                            break
                        file.write(chunk)
                        offset += len(chunk)
                        if offset >= report_at:
                            print(f'{target.name}: {offset / 1e6:.1f}/{expected / 1e6:.1f} MB', flush=True)
                            report_at = offset + 32 * 1024 ** 2
            assert partial.stat().st_size == expected, 'Incomplete download'
            assert checksum(partial) == item['sha256'], f'Download hash mismatch: {partial}'
            partial.rename(target)
            print('SHA256 verified:', target.name, flush=True)
            return
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
            if attempt == 49 or 'hash mismatch' in str(error):
                raise
            time.sleep(3)


for item in manifest:
    download(item)
print('All official CUDA wheels cached and verified.', flush=True)
