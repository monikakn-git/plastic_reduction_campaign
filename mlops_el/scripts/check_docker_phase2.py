"""Record actual container identity, isolation, dependencies and HTTP inference evidence."""
import argparse,hashlib,json,subprocess,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def docker(*args):
    return subprocess.check_output(['sh',str(ROOT/'scripts/docker_local.sh'),*args],cwd=ROOT,text=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8011);args=parser.parse_args()
    cid=docker('compose','ps','-q','detector').strip();assert cid,'No running detector container'
    record=json.loads(docker('inspect',cid))[0];image=json.loads(docker('image','inspect',record['Image']))[0]
    assert record['State']['Running'] and record['State']['Health']['Status']=='healthy'
    assert record['Config']['User']=='10001:10001' and record['HostConfig']['ReadonlyRootfs']
    mounts=record['Mounts'];assert len(mounts)==1 and mounts[0]['Destination']=='/models' and not mounts[0]['RW']
    bindings=record['HostConfig']['PortBindings']['8000/tcp'];assert bindings==[{'HostIp':'127.0.0.1','HostPort':str(args.port)}]
    base=f'http://127.0.0.1:{args.port}'
    ready=json.load(urllib.request.urlopen(base+'/ready',timeout=10))
    release=json.loads((ROOT/'releases/yolo11m-demo-v3/release.json').read_text())
    rows=json.loads((ROOT/'data/prepared/plitterstreet-labels-v3/splits.json').read_text());row=next(r for r in rows if r['split']=='train')
    raw=(ROOT/'data/prepared/plitterstreet-labels-v3/images/train'/row['file_name']).read_bytes()
    request=urllib.request.Request(base+'/predict',data=raw,headers={'Content-Type':'image/jpeg'})
    started=time.perf_counter()
    with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response);status=response.status
    elapsed=(time.perf_counter()-started)*1000
    assert ready['model_sha256']==result['model']['sha256']==release['weights_sha256']
    assert result['model']['registry_version']==release['registry_version'] and result['detections']
    for detection in result['detections']:
        assert detection['label']==release['classes'][detection['class_id']]
        assert .25<=detection['confidence']<=1
        x1,y1,x2,y2=detection['bbox_xyxy']
        assert 0<=x1<=x2<=result['width'] and 0<=y1<=y2<=result['height']
    try:
        urllib.request.urlopen(urllib.request.Request(base+'/predict',data=b'bad image',headers={'Content-Type':'image/jpeg'}),timeout=10)
        raise AssertionError('Malformed upload accepted')
    except urllib.error.HTTPError as exc:assert exc.code==422
    out=ROOT/'reports/phase2';out.mkdir(exist_ok=True)
    (out/'container_requirements.txt').write_text(docker('exec',cid,'python','-m','pip','freeze'))
    proof={'status':'passed','deployment_type':'docker_colima_arm64_cpu','address':base,'container_id':cid,'image_id':image['Id'],'image_architecture':image['Architecture'],'image_os':image['Os'],'image_size_bytes':image['Size'],'health':record['State']['Health']['Status'],'non_root_user':record['Config']['User'],'read_only_root':True,'model_mount_read_only':True,'port_bindings':bindings,'memory_limit_bytes':record['HostConfig']['Memory'],'pids_limit':record['HostConfig']['PidsLimit'],'security_opt':record['HostConfig']['SecurityOpt'],'cap_drop':record['HostConfig']['CapDrop'],'prediction_http_status':status,'invalid_image_status':422,'sample_split':'train','sample_image_id':row['image_id'],'sample_sha256':hashlib.sha256(raw).hexdigest(),'detections':len(result['detections']),'single_request_ms':elapsed,'latency_scope':'One smoke request, not a throughput benchmark','model':result['model'],'production_approved':False,'docker_version':json.loads(docker('version','--format','{{json .}}')),'verified_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    (out/'docker_deployment.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
