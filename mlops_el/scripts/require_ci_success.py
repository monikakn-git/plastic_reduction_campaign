"""Require a successful CI run for this exact commit before local deployment."""
import json,os,urllib.request
from pathlib import Path

def main():
    repo=os.environ['GITHUB_REPOSITORY'];sha=os.environ['GITHUB_SHA'];token=os.environ['GITHUB_TOKEN']
    url=f'https://api.github.com/repos/{repo}/actions/workflows/phase2-ci.yml/runs?head_sha={sha}&status=completed&per_page=100'
    request=urllib.request.Request(url,headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(request,timeout=30) as response:runs=json.load(response)['workflow_runs']
    eligible=[r for r in runs if r['head_sha']==sha and r['head_branch']=='main' and r['event'] in ('push','workflow_dispatch')]
    if not eligible:raise SystemExit('No completed CI run exists for this exact main-branch commit; deploy refused')
    latest=max(eligible,key=lambda r:r['id'])
    if latest['conclusion']!='success':raise SystemExit('Latest completed CI for this commit did not succeed; deploy refused')
    proof={'ci_run_id':latest['id'],'head_sha':sha,'conclusion':'success','url':latest['html_url']}
    out=Path('.local');out.mkdir(exist_ok=True);(out/'phase2-ci-gate.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))
if __name__=='__main__':main()
