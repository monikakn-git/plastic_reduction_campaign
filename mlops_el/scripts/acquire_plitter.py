"""Acquire 10,000 original pLitterStreet images using verified ZIP byte ranges.

Only selected image entries are stored. No recompression, augmentation, or
thumbnail substitution is used to reach the target count.
"""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import math
import json
from pathlib import Path
import random
import re
import shutil
import struct
import threading
import time
import zipfile
import zlib

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/plitterstreet'
REPORT = ROOT / 'reports/phase1'
URL = 'https://zenodo.org/records/8288500/files/images.zip'
SIZE = 13728630001
LOCAL = threading.local()
CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
BLOCK_BYTES = 16*1024**2
MAX_BLOCKS = 16
COOLDOWN_UNTIL = 0.0
NETWORK_ABORT = threading.Event()
NETWORK_STOP_REASON = ''


def write_bytes(path, content):
    """Replace metadata atomically, without modifying any linked DVC cache file."""
    temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path, value):
    write_bytes(path,json.dumps(value,indent=2).encode())


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def fetch_range(start, stop):
    """Fetch [start, stop), refusing full-archive fallbacks and wrong ranges."""
    global COOLDOWN_UNTIL, NETWORK_STOP_REASON
    for attempt in range(4):
        try:
            if NETWORK_ABORT.is_set():
                raise requests.ConnectionError('Acquisition stopped after a systemic network failure')
            with CACHE_LOCK:
                delay=max(0,COOLDOWN_UNTIL-time.monotonic())
            if delay:
                time.sleep(delay)
            with session().get(URL, headers={'Range': f'bytes={start}-{stop-1}'},
                               stream=True, timeout=(20, 120)) as response:
                if response.status_code==429:
                    try:
                        delay=max(60,float(response.headers.get('Retry-After','60')))
                    except ValueError:
                        try:
                            deadline=parsedate_to_datetime(response.headers['Retry-After'])
                            delay=max(60,(deadline-datetime.now(timezone.utc)).total_seconds())
                        except (KeyError,TypeError,ValueError):
                            delay=120
                    with CACHE_LOCK:
                        COOLDOWN_UNTIL=max(COOLDOWN_UNTIL,time.monotonic()+delay)
                response.raise_for_status()
                expected = f'bytes {start}-{stop-1}/{SIZE}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Server did not honour requested range: {response.status_code}, {response.headers.get("Content-Range")}')
                content = response.content
                if len(content) != stop-start:
                    raise ValueError('Truncated range response')
                return content
        except (requests.RequestException, ValueError) as error:
            message=str(error).lower()
            if isinstance(error,requests.exceptions.SSLError) or any(token in message for token in
                    ('nameresolutionerror','failed to resolve','nodename nor servname','acquisition stopped')):
                NETWORK_STOP_REASON=NETWORK_STOP_REASON or str(error)
                NETWORK_ABORT.set()
                raise
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def cached_block(number):
    """Share bounded read-ahead blocks to avoid one server request per image."""
    with CACHE_LOCK:
        future=CACHE.get(number)
        owner=future is None
        if owner:
            future=Future()
            CACHE[number]=future
        CACHE.move_to_end(number)
    if owner:
        try:
            content=fetch_range(number*BLOCK_BYTES,min(SIZE,(number+1)*BLOCK_BYTES))
            future.set_result(content)
        except Exception as error:
            future.set_exception(error)
            with CACHE_LOCK:
                CACHE.pop(number,None)
            raise
        with CACHE_LOCK:
            for key in list(CACHE):
                if len(CACHE)<=MAX_BLOCKS:
                    break
                if CACHE[key].done() and key!=number:
                    CACHE.pop(key)
    return future.result()


def byte_range(start, stop):
    parts=[]
    position=start
    while position<stop:
        number=position//BLOCK_BYTES
        block=cached_block(number)
        offset=position-number*BLOCK_BYTES
        take=min(stop-position,len(block)-offset)
        if take<=0:
            raise ValueError('Invalid range in archive block')
        parts.append(block[offset:offset+take])
        position+=take
    return b''.join(parts)


class RemoteZip(io.RawIOBase):
    """Seekable metadata reader; never downloads the full ZIP implicitly."""
    def __init__(self):
        self.position = 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else SIZE+offset
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position
    def read(self, size=-1):
        stop = SIZE if size < 0 else min(SIZE, self.position+size)
        if stop <= self.position:
            return b''
        if stop-self.position > 32*1024**2:
            raise ValueError('Refusing an unexpectedly large ZIP metadata read')
        content = byte_range(self.position, stop)
        self.position = stop
        return content


def archive_index():
    path = RAW/'archive_index.json'
    if path.exists():
        return json.loads(path.read_text())
    with zipfile.ZipFile(RemoteZip()) as archive:
        entries = [{'name': x.filename, 'offset': x.header_offset, 'compressed_bytes': x.compress_size,
                    'bytes': x.file_size, 'crc32': x.CRC, 'method': x.compress_type}
                   for x in archive.infolist() if not x.is_dir()]
    write_json(path,entries)
    return entries


def image_group(record):
    # All frames sharing a source-video stem stay together. Unique citizen photos
    # have unique stems; source dataset ID avoids collisions across regions.
    stem = Path(record['file_name']).stem
    return f"{record.get('dataset_id','unknown')}:{re.sub(r'[_-][0-9]+$', '', stem)}"


def acquire_image(pair, min_free_gb, direct=False):
    record, entry = pair
    target = RAW/'images'/record['file_name']
    try:
        if not target.exists():
            if NETWORK_ABORT.is_set():
                raise requests.ConnectionError('Not downloaded: acquisition stopped after a network failure; rerun to resume')
            if shutil.disk_usage(ROOT).free < min_free_gb*1024**3:
                raise OSError('Free-space safety floor reached; no images dropped or resized')
            # Include the local header and enough room for normal names/extras.
            # Rare unusually long extra fields are handled by a second exact read.
            start = entry['offset']
            stop = min(SIZE, start+entry['compressed_bytes']+8192)
            reader=fetch_range if direct else byte_range
            block = reader(start, stop)
            header = struct.unpack('<4s5H3I2H', block[:30])
            if header[0] != b'PK\x03\x04' or header[2] & 1:
                raise ValueError('Invalid/encrypted ZIP member')
            skip = 30+header[-2]+header[-1]
            local_name = block[30:30+header[-2]].decode('utf-8')
            if local_name != entry['name']:
                raise ValueError('Local and central ZIP names differ')
            if len(block) < skip+entry['compressed_bytes']:
                block = reader(start, start+skip+entry['compressed_bytes'])
            compressed = block[skip:skip+entry['compressed_bytes']]
            content = zlib.decompress(compressed, -15) if entry['method'] == 8 else compressed if entry['method'] == 0 else None
            if content is None or len(content) != entry['bytes'] or zlib.crc32(content) != entry['crc32']:
                raise ValueError('ZIP member length or CRC mismatch')
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            temp = target.with_suffix(target.suffix+'.part')
            temp.write_bytes(content)
            temp.replace(target)
        content = target.read_bytes()
        if len(content) != entry['bytes'] or zlib.crc32(content) != entry['crc32']:
            raise ValueError('Existing file differs from official ZIP member')
        with Image.open(io.BytesIO(content)) as image:
            orientation = image.getexif().get(274, 1)
            dimensions = image.size[::-1] if orientation in (5,6,7,8) else image.size
            image.verify()
        if dimensions != (record['width'], record['height']):
            raise ValueError(f'Annotation/image dimension mismatch: {dimensions}')
        return {'image_id': record['id'], 'file_name': record['file_name'],
                'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
                'crc32': entry['crc32'], 'width': dimensions[0], 'height': dimensions[1],
                'group_id': image_group(record), 'exif_orientation': orientation}
    except Exception as error:
        return {'image_id': record['id'], 'file_name': record['file_name'], 'error': str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-images', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-free-gb', type=float, default=10)
    parser.add_argument('--index-only', action='store_true')
    args = parser.parse_args()
    (RAW/'images').mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    annotations_zip = ROOT/'data/downloads/plitter-annotations.zip'
    annotations_zip.parent.mkdir(parents=True,exist_ok=True)
    if not annotations_zip.exists():
        response=session().get('https://zenodo.org/records/8288500/files/annotations.zip',timeout=(20,120))
        response.raise_for_status()
        write_bytes(annotations_zip,response.content)
    if not (RAW/'record.json').exists():
        response=session().get('https://zenodo.org/api/records/8288500',timeout=60)
        response.raise_for_status()
        write_json(RAW/'record.json',response.json())
    if hashlib.md5(annotations_zip.read_bytes()).hexdigest() != '2484e076b1e9d66036b0a15e9928692f':
        raise SystemExit('Official annotation archive MD5 does not match')
    with zipfile.ZipFile(annotations_zip) as archive:
        original = archive.read('annotations/10_class_splits/annotations_all.json')
    write_bytes(RAW/'annotations_all.json',original)
    data = json.loads(original)
    index = archive_index()
    by_name = {}
    for item in index:
        name = Path(item['name']).name
        if name in by_name:
            raise ValueError(f'Duplicate ZIP basename: {name}')
        by_name[name] = item
    candidates = sorted(data['images'], key=lambda x:x['id'])
    random.Random(args.seed).shuffle(candidates)
    if not 0 < args.target_images <= len(candidates):
        raise ValueError('Target exceeds available source images')
    if any(Path(i['file_name']).name != i['file_name'] for i in candidates):
        raise ValueError('Expected safe, flat source filenames')
    if any(i['file_name'] not in by_name for i in candidates):
        raise ValueError('Annotated images missing from official ZIP index')
    selection_path=RAW/'selection.json'
    previous_selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if previous_selection and (previous_selection.get('seed')!=args.seed or
            previous_selection.get('target_images')!=args.target_images or
            previous_selection.get('source_annotations_sha256')!=hashlib.sha256(original).hexdigest()):
        raise SystemExit('Existing selection differs from requested source/seed/count; review explicitly before replacing it')
    lookup={i['id']:i for i in candidates}
    selected=[lookup[i] for i in previous_selection['image_ids']] if previous_selection else candidates[:args.target_images]
    if len(selected)!=args.target_images or len({i['id'] for i in selected})!=args.target_images:
        raise SystemExit('Persisted selection has invalid image IDs or count')
    projected = sum(by_name[i['file_name']]['bytes'] for i in selected)
    info = {'source':'https://zenodo.org/records/8288500','images_url':URL,
            'archive_md5_as_published':'02382a3b9620ae7b086ff4c7fd11891d',
            'archive_integrity_method':'Selected entries validated with ZIP CRC32 and recorded SHA256; full archive MD5 is not claimed verified',
            'annotations_archive_md5_verified':'2484e076b1e9d66036b0a15e9928692f',
            'source_annotations_sha256':hashlib.sha256(original).hexdigest(),
            'source_image_count':len(candidates),'target_images':args.target_images,
            'selection_seed':args.seed,'projected_selected_image_bytes':projected,
            'licence':'CC BY 4.0 per official Zenodo API metadata',
            'dataset_approval':'pending faculty approval'}
    write_json(RAW/'source.json',info)
    print(json.dumps(info,indent=2),flush=True)
    if args.index_only:
        return
    existing_bytes=sum((RAW/'images'/i['file_name']).stat().st_size for i in selected if (RAW/'images'/i['file_name']).exists())
    if shutil.disk_usage(ROOT).free < projected-existing_bytes+args.min_free_gb*1024**3:
        raise SystemExit('Insufficient free space for original selected images plus OS headroom; no quality reduction made')
    manifest,failures=[],[]
    # Interleave ordered archive regions so each concurrent transfer reads ahead
    # locally. Selection itself remains the original seeded 10,000-image sample.
    ordered=sorted(selected,key=lambda i:by_name[i['file_name']]['offset'])
    lane_size=math.ceil(len(ordered)/args.workers)
    lanes=[ordered[n:n+lane_size] for n in range(0,len(ordered),lane_size)]
    schedule=[lane[n] for n in range(lane_size) for lane in lanes if n<len(lane)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(acquire_image,(i,by_name[i['file_name']]),args.min_free_gb) for i in schedule]
        for count,future in enumerate(as_completed(tasks),1):
            result=future.result()
            (failures if 'error' in result else manifest).append(result)
            if count%100==0 or count==len(tasks):
                print(f'Checked {count}/{len(tasks)}; verified={len(manifest)}, failed={len(failures)}',flush=True)
                (REPORT/'acquisition_progress.json').write_text(json.dumps({'dataset':'plitterstreet','target':args.target_images,'verified':len(manifest),'failed':len(failures),'failures':failures,'complete':False},indent=2))
    # A failed first transfer is not a reason to exclude an image. Retry the
    # exact same selected files with less concurrency after the main pass.
    selected_by_id={i['id']:i for i in selected}
    for retry_round in range(2):
        if not failures or NETWORK_ABORT.is_set():
            break
        print(f'Retrying {len(failures)} failed selected images, pass {retry_round+1}',flush=True)
        time.sleep(10)
        retry_failures=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(acquire_image,(selected_by_id[f['image_id']],by_name[f['file_name']]),args.min_free_gb)
                   for f in failures]
            for future in as_completed(tasks):
                result=future.result()
                (retry_failures if 'error' in result else manifest).append(result)
        failures=retry_failures
        print(f'After retry: verified={len(manifest)}, failed={len(failures)}',flush=True)
    exclusions=previous_selection.get('exact_duplicate_exclusions',[])
    if not failures and not NETWORK_ABORT.is_set():
        # Keep the lowest source ID for exact byte-identical photos. Source
        # annotation variants remain in annotations_all.json for review.
        seen={}
        retained=[]
        for item in sorted(manifest,key=lambda m:m['image_id']):
            if item['sha256'] in seen:
                exclusions.append({'removed_image_id':item['image_id'],
                    'kept_image_id':seen[item['sha256']]['image_id'],
                    'file_name':item['file_name'],'sha256':item['sha256'],
                    'reason':'Exact byte-identical photo; retain lowest source ID; original annotation variants preserved'})
            else:
                seen[item['sha256']]=item
                retained.append(item)
        used_ids={i['id'] for i in selected}|{e['removed_image_id'] for e in exclusions}
        for candidate in candidates:
            if len(retained)==args.target_images:
                break
            if candidate['id'] in used_ids:
                continue
            used_ids.add(candidate['id'])
            print(f"Replacing an exact duplicate with source image {candidate['id']}",flush=True)
            result=acquire_image((candidate,by_name[candidate['file_name']]),args.min_free_gb,direct=True)
            if 'error' in result:
                failures.append(result)
                break
            if result['sha256'] in seen:
                exclusions.append({'removed_image_id':result['image_id'],
                    'kept_image_id':seen[result['sha256']]['image_id'],
                    'file_name':result['file_name'],'sha256':result['sha256'],
                    'reason':'Replacement candidate was also an exact duplicate'})
            else:
                seen[result['sha256']]=result
                retained.append(result)
        manifest=retained
        selected=[lookup[m['image_id']] for m in manifest]
        if not failures and len(manifest)==args.target_images:
            for item in exclusions:
                source=RAW/'images'/item['file_name']
                if source.exists():
                    destination=RAW/'excluded_exact_duplicates'/item['file_name']
                    destination.parent.mkdir(exist_ok=True)
                    source.replace(destination)
            selection={'seed':args.seed,'target_images':args.target_images,
                'source_annotations_sha256':hashlib.sha256(original).hexdigest(),
                'image_ids':sorted(i['id'] for i in selected),
                'exact_duplicate_exclusions':exclusions,
                'policy':'Seeded initial selection; exact duplicate photos canonicalised to lowest source ID; backfill from unused candidates in the same seeded order'}
            write_json(selection_path,selection)
            info['selection_manifest_sha256']=hashlib.sha256(selection_path.read_bytes()).hexdigest()
    duplicates=len(manifest)-len({m['sha256'] for m in manifest})
    selected_ids={i['id'] for i in selected}
    for image in selected:
        image['group_id']=image_group(image)
    subset={**data,'images':sorted(selected,key=lambda x:x['id']),
            'annotations':[a for a in data['annotations'] if a['image_id'] in selected_ids]}
    annotation_bytes=json.dumps(subset,indent=2).encode()
    write_bytes(RAW/'annotations.json',annotation_bytes)
    write_json(RAW/'manifest.json',sorted(manifest,key=lambda x:x['image_id']))
    info['annotations_sha256']=hashlib.sha256(annotation_bytes).hexdigest()
    info['downloaded_at_utc']=datetime.now(timezone.utc).isoformat()
    write_json(RAW/'source.json',info)
    summary={'dataset':'plitterstreet','status':'complete' if not failures and not duplicates and len(manifest)==args.target_images else 'incomplete',
             'expected_images':args.target_images,'verified_images':len(manifest),
             'unique_image_hashes':len({m['sha256'] for m in manifest}),
             'annotations':len(subset['annotations']),'source_categories':len(data['categories']),
             'downloaded_image_bytes':sum(m['bytes'] for m in manifest),'failures':failures,
             'systemic_network_abort':NETWORK_ABORT.is_set(),'network_stop_reason':NETWORK_STOP_REASON,
             'duplicate_image_count':duplicates,'annotations_sha256':info['annotations_sha256']}
    summary['excluded_exact_duplicate_copies']=len(exclusions)
    (REPORT/'acquisition.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if failures or duplicates or len(manifest)!=args.target_images:
        raise SystemExit('Acquisition incomplete: errors or duplicates require review. No baseline started.')


if __name__=='__main__':
    main()
