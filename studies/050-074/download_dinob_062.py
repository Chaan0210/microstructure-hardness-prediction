"""Authorized official checkpoint download, confined to project vendor folder."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,hashlib,urllib.request,time,shutil
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dino_frozen_037 as base
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
DEST=_repo_path(ROOT, 'vendor/dinov2-base-062');DEST.mkdir(exist_ok=False)
assert URL in (base.SOURCE/'README.md').read_text()
card=(base.SOURCE/'MODEL_CARD.md').read_text();assert '**License:** Apache License 2.0' in card
for n in ('LICENSE','MODEL_CARD.md'):shutil.copyfile(_repo_io(base.SOURCE/n),_repo_io(_repo_path(DEST, n)))
start=time.perf_counter();sha=hashlib.sha256();size=0;partial=_repo_path(DEST, 'dinov2_vitb14_pretrain.pth.partial')
with urllib.request.urlopen(URL,timeout=60) as response:
 assert response.status==200 and response.geturl()==URL
 expected=int(response.headers.get('Content-Length','0'));assert expected==0 or 300_000_000<expected<400_000_000
 with partial.open('xb') as file:
  while True:
   block=response.read(2**20)
   if not block:break
   file.write(block);sha.update(block);size+=len(block);assert size<400_000_000
   if size%(32*2**20)==0:print(json.dumps({'downloaded_bytes':size,'seconds':time.perf_counter()-start}),flush=True)
assert size==expected if expected else size>300_000_000
path=_repo_path(DEST, 'dinov2_vitb14_pretrain.pth');partial.rename(path)
report={'url':URL,'bytes':size,'http_content_length':expected,'sha256':sha.hexdigest(),'seconds':time.perf_counter()-start,'official_source_dir':str(base.SOURCE.relative_to(ROOT)),'code_commit':base.PROVENANCE.get('commit','see source_dir'),'license':'Apache-2.0 frompinnedofficialoriginalDINOv2MODEL_CARD andLICENSE','license_sha256':hashlib.sha256((_repo_path(DEST, 'LICENSE')).read_bytes()).hexdigest(),'model_card_sha256':hashlib.sha256((_repo_path(DEST, 'MODEL_CARD.md')).read_bytes()).hexdigest(),'model':'originalnonregisterdinov2_vitb14','packages_installed':0,'cost':0,'hash_interpretation':'Locallycomputedintegrityhash,notpublisher-signedchecksum;trustedHTTPSofficialURLandstrictweightsloadwillbeverified.'}
with (_repo_path(DEST, 'provenance.json')).open('x') as file:json.dump(report,file,indent=2);file.write('\n')
print(json.dumps(report),flush=True)
