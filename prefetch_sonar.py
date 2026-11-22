"""Download original SONAR assets into fairseq2's cache with persistent resume."""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

root = Path(__file__).resolve().parent
cache = root / '.cache/fairseq2/assets'
cache.mkdir(parents=True, exist_ok=True)
manifest = []
for filename in ['sonar_text_encoder.pt', 'sonar_text_decoder.pt', 'sentencepiece.source.256000.model']:
    url = 'https://dl.fbaipublicfiles.com/SONAR/' + filename
    key = hashlib.sha1(url.encode()).hexdigest()[:24]
    directory = cache / key
    target = directory / filename
    partial = cache / (key + '.partial')
    expected = None
    if not target.exists():
        for attempt in range(50):
            offset = partial.stat().st_size if partial.exists() else 0
            if expected is not None and offset == expected:
                break
            print(f'{filename}: resume {offset / 1e6:.1f} MB, attempt {attempt + 1}', flush=True)
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urlopen(Request(url, headers=headers), timeout=90) as response:
                    if offset:
                        assert response.status == 206, 'Resume range not honored'
                        assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    expected = offset + int(response.headers['Content-Length'])
                    with partial.open('ab') as file:
                        report_at = offset + 128 * 1024 ** 2
                        while True:
                            chunk = response.read(1024 ** 2)
                            if not chunk:
                                break
                            file.write(chunk)
                            offset += len(chunk)
                            if offset >= report_at:
                                print(f'{filename}: {offset / 1e6:.1f}/{expected / 1e6:.1f} MB', flush=True)
                                report_at = offset + 128 * 1024 ** 2
                assert partial.stat().st_size == expected, 'Incomplete transfer'
                break
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
                if attempt == 49:
                    raise
                time.sleep(3)
        if zipfile.is_zipfile(partial):
            with zipfile.ZipFile(partial) as archive:
                assert archive.testzip() is None, 'Checkpoint ZIP CRC error'
        stage = cache / (key + '.prefetch')
        stage.mkdir(exist_ok=True)
        partial.replace(stage / filename)
        assert not directory.exists(), 'Unexpected concurrent cache creation'
        stage.replace(directory)
    digest = hashlib.sha256()
    with target.open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 ** 2), b''):
            digest.update(chunk)
    manifest.append({'url': url, 'path': str(target), 'bytes': target.stat().st_size,
                     'sha256_local': digest.hexdigest(),
                     'validation': 'HTTPS official asset; HTTP byte length; ZIP CRC when applicable; local SHA256 recorded'})
    (root / 'results/model_download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Asset ready:', filename, target.stat().st_size, flush=True)
print('All original SONAR assets ready.', flush=True)
