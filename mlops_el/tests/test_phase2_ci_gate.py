"""Deployment must use successful CI for its exact source commit."""
import json
import pytest
from scripts.require_ci_success import main

class Response:
    def __init__(self,runs):self.runs=runs
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps({'workflow_runs':self.runs}).encode()

def setup(monkeypatch,tmp_path,runs):
    monkeypatch.chdir(tmp_path)
    for key,value in {'GITHUB_REPOSITORY':'owner/private','GITHUB_SHA':'a'*40,'GITHUB_TOKEN':'test-placeholder'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr('scripts.require_ci_success.urllib.request.urlopen',lambda request,timeout:Response(runs))

def run_record(identifier=1,sha='a'*40,branch='main',conclusion='success'):
    return {'id':identifier,'head_sha':sha,'head_branch':branch,'event':'push','conclusion':conclusion,'html_url':f'https://github.com/owner/private/actions/runs/{identifier}'}

def test_successful_exact_commit_produces_gate_receipt(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,[run_record()]);main()
    receipt=json.loads((tmp_path/'.local/phase2-ci-gate.json').read_text());assert receipt['head_sha']=='a'*40 and receipt['conclusion']=='success'

@pytest.mark.parametrize('runs', [[],[run_record(sha='b'*40)],[run_record(branch='other')],[run_record(conclusion='failure')],[run_record(1),run_record(2,conclusion='failure')]])
def test_absent_mismatched_or_latest_failed_ci_blocks_deploy(monkeypatch,tmp_path,runs):
    setup(monkeypatch,tmp_path,runs)
    with pytest.raises(SystemExit):main()
    assert not (tmp_path/'.local/phase2-ci-gate.json').exists()
