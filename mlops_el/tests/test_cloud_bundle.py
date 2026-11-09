import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cloud'))
from runner import safe_name,unpack,sha,save_checkpoint_generation,verify_latest


def test_archive_paths_and_links_are_rejected(tmp_path):
    for path in ('../escape','/absolute','dataset/../../escape','a\\b'):
        with pytest.raises(ValueError):safe_name(path)
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as f:
        member=tarfile.TarInfo('link');member.type=tarfile.SYMTYPE;member.linkname='../outside';f.addfile(member)
    with pytest.raises(ValueError,match='regular'):unpack(archive,tmp_path/'extract',sha(archive))


def test_real_scope_counts_and_checksums_on_extract(tmp_path):
    archive=tmp_path/'bundle.tar';entries=[]
    with tarfile.open(archive,'w') as f:
        for split,count in [('train',256),('val',32)]:
            for i in range(count):
                for kind,ext,payload in [('images','jpg',b'fixture image'),('labels','txt',b'')]:
                    name=f'dataset/{kind}/{split}/{i}.{ext}'
                    m=tarfile.TarInfo(name);m.size=len(payload);f.addfile(m,io.BytesIO(payload))
                    entries.append({'path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        data=json.dumps({'scope':'benchmark','train_images':256,'val_images':32,'files':entries}).encode()
        m=tarfile.TarInfo('manifest.json');m.size=len(data);f.addfile(m,io.BytesIO(data))
    result=unpack(archive,tmp_path/'verified',sha(archive));assert result['scope']=='benchmark'
    with pytest.raises(ValueError,match='checksum'):unpack(archive,tmp_path/'wrong','0'*64)


def test_checkpoint_generations_are_verified_and_bounded(tmp_path):
    weights=tmp_path/'weights';weights.mkdir();last=weights/'last.pt';best=weights/'best.pt'
    persistent=tmp_path/'persist';metadata={'mode':'benchmark','run_id':'fixture','bundle_fingerprint':'fixture'}
    for epoch in range(3):
        last.write_bytes(f'optimizer-checkpoint-{epoch}'.encode());best.write_bytes(b'best')
        save_checkpoint_generation(SimpleNamespace(epoch=epoch,last=last,best=best),persistent,metadata)
    checkpoint,state=verify_latest(persistent)
    assert state['epoch']==3 and checkpoint.read_bytes()==last.read_bytes()
    assert len(list((persistent/'checkpoints').glob('epoch-*')))==2
    checkpoint.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):verify_latest(persistent)
