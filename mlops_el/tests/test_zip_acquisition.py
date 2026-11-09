"""Verify original JPEG extraction and CRC rejection without network access."""
import importlib.util
import io
from pathlib import Path
import zipfile
import threading
import pytest
from PIL import Image

spec=importlib.util.spec_from_file_location('acquire',Path(__file__).parents[1]/'scripts/acquire_plitter.py')
acquire=importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


def fixture_entry():
    image=io.BytesIO()
    Image.new('RGB',(100,80),(20,60,90)).save(image,format='JPEG')
    original=image.getvalue()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('images/test.jpg',original)
    content=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        item=archive.getinfo('images/test.jpg')
        entry={'name':item.filename,'offset':item.header_offset,'compressed_bytes':item.compress_size,
               'bytes':item.file_size,'crc32':item.CRC,'method':item.compress_type}
    return original,content,entry


def test_original_bytes_preserved(monkeypatch,tmp_path):
    original,content,entry=fixture_entry()
    monkeypatch.setattr(acquire,'RAW',tmp_path)
    monkeypatch.setattr(acquire,'byte_range',lambda a,b:content[a:b])
    (tmp_path/'images').mkdir()
    result=acquire.acquire_image(({'id':1,'file_name':'test.jpg','width':100,'height':80},entry),0)
    assert 'error' not in result
    assert (tmp_path/'images/test.jpg').read_bytes()==original


def test_bad_crc_rejected(monkeypatch,tmp_path):
    _,content,entry=fixture_entry()
    entry['crc32']^=1
    monkeypatch.setattr(acquire,'RAW',tmp_path)
    monkeypatch.setattr(acquire,'byte_range',lambda a,b:content[a:b])
    (tmp_path/'images').mkdir()
    result=acquire.acquire_image(({'id':1,'file_name':'test.jpg','width':100,'height':80},entry),0)
    assert 'error' in result
    assert not (tmp_path/'images/test.jpg').exists()


def test_dns_outage_stops_new_network_requests(monkeypatch):
    calls=[]
    class Unreachable:
        def get(self,*args,**kwargs):
            calls.append(args)
            raise acquire.requests.ConnectionError('NameResolutionError: failed to resolve dataset host')
    monkeypatch.setattr(acquire,'session',lambda:Unreachable())
    monkeypatch.setattr(acquire,'NETWORK_ABORT',threading.Event())
    monkeypatch.setattr(acquire,'NETWORK_STOP_REASON','')
    with pytest.raises(acquire.requests.ConnectionError):
        acquire.fetch_range(0,1024)
    with pytest.raises(acquire.requests.ConnectionError):
        acquire.fetch_range(1024,2048)
    assert len(calls)==1
    assert acquire.NETWORK_ABORT.is_set()
