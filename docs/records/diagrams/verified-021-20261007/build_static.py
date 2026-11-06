from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
from html import escape
import re
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
TEMPLATE=(_repo_path(ROOT, '.agents/skills/diagram-design/assets/template.html')).read_text()
P='#f5f5f5';I='#2d3142';M='#4f5d75';A='#eb6c36'
def txt(x,y,s,cls='body'):
 return f'<text x="{x}" y="{y}" class="{cls}">{escape(s)}</text>'
def node(id,x,y,w,h,title,lines,accent=False):
 r=f'<g class="node" data-node="{id}"><rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{ "#fcf0e9" if accent else "#fff"}" stroke="{A if accent else I}" stroke-width="{2 if accent else 1}"/>'
 r+=txt(x+24,y+36,title,'name')
 for j,s in enumerate(lines):
  cls='body' if re.search('[가-힣]',s) else 'tech'
  r+=txt(x+24,y+68+j*24,s,cls)
 return r+'</g>'
def arrow(d):return f'<path class="connector" d="{d}" fill="none" stroke="{M}" stroke-width="1.5" marker-end="url(#arrow)"/>'
def make(slug,title,desc,H,content,foot):
 W=1600
 svg=f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc"><title id="{slug}-title">{title}</title><desc id="{slug}-desc">{desc}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{M}"/></marker></defs><rect width="100%" height="100%" fill="{P}"/>'
 svg+=txt(40,52,'MICROSTRUCTURE / VERIFIED CODE / 2026.10.07','eyebrow-svg')+txt(40,108,title,'title-svg')+txt(40,140,desc,'body')+content
 svg+=f'<line x1="40" y1="{H-76}" x2="1560" y2="{H-76}" stroke="#bfc0c0"/>'
 svg+=txt(40,H-44,foot,'body')+txt(40,H-20,'Diagram Design v2.6 · d137637 · 6/8 grouped nodes · offline system-font fallback · PNG 2×','footer-svg')+'</svg>'
 html=re.sub(r'<link href=.*?rel="stylesheet">','',TEMPLATE,flags=re.S)
 html=html.replace('lang="en"','lang="ko"').replace('<title>Diagram</title>',f'<title>{title}</title>')
 html=re.sub(r'<p class="eyebrow">.*?</h1>','',html,flags=re.S)
 html=re.sub(r'<svg viewBox=.*?</svg>',svg,html,flags=re.S)
 html=html.replace('min-width: 960px','min-width: 1600px').replace('max-width: 1200px','max-width: 1600px').replace('padding: 3rem 2rem','padding: 0')
 css="""
 .frame { min-width:0; max-width:100%; } svg { width:1600px; }
 svg text { fill:#2d3142; font-family:'Geist','Noto Sans KR','Apple SD Gothic Neo',Arial,sans-serif; }
 .title-svg {font-family:'Instrument Serif','Noto Serif KR','AppleMyungjo',Georgia,serif;font-size:40px;font-weight:400;}
 .name {font-size:16px;font-weight:600;} .body {font-size:16px;font-weight:400;}
 .tech {font-family:'Geist Mono',Menlo,monospace;font-size:12px;}
 .eyebrow-svg {font-family:'Geist Mono',Menlo,monospace;font-size:12px;letter-spacing:2px;}
 .footer-svg {font-family:'Geist Mono',Menlo,monospace;font-size:11px;fill:#4f5d75;}
 """
 html=html.replace('</style>',css+'</style>')
 (_repo_path(OUT, f'{slug}.html')).write_text(html)

c=''.join([arrow('M 296 340 H 328 Q 336 340 336 332 V 308 Q 336 300 344 300 H 376'),arrow('M 296 460 H 328 Q 336 460 336 468 V 656 Q 336 664 344 664 H 376'),arrow('M 664 664 H 720'),arrow('M 1032 664 H 1088'),arrow('M 1032 300 H 1120'),arrow('M 1320 552 V 440')])
c+=node('image',40,280,256,240,'입력 · 동일 이미지', ['256 × 256 · grayscale','원본은 한 장의 미세조직 이미지','각 경로가 별도로 전처리','학습·시험의 ID 순서 보존'])
c+=node('018',376,200,656,240,'018 · 보정 + short-B + 잔차를 모두 포함', ['raw = .3 KRR26 + .3 CNN(mean3) + .2 S(mean3) + .2 B(mean3)','cal(raw) = 194.1655162532 + 1.1103548755 × (raw − 194.0592588401)','018 = .9 cal(raw) + .1 shortB + mean3(weighted residual)','잔차: physics57 + DINO-B spatial528 → 585차원','정확한 내부 순서와 각 층은 018 상세 그림 참조'],True)
c+=node('prepL',376,552,288,224,'L 경로 전처리', ['BICUBIC resize → 504 × 504','1 channel → replicated RGB','/255 → ImageNet normalization','μ=[.485,.456,.406]','σ=[.229,.224,.225]'])
c+=node('dinoL',720,552,312,224,'동결 DINOv2-L / ViT-L/14',['24 blocks · width 1024','1296 patches · 36 × 36','layers = [11,17,23] (0-based)','각 층: normalized CLS + mean + std','3 layers × 3 pools × 1024 = 9216'])
c+=node('headL',1088,552,456,224,'L 헤드 3개 → 예측 평균', ['9216 → StandardScaler → 64 → 1','GELU · Dropout(.25), eval에서 비활성','target scale을 복원한 뒤 3 seed 평균','seeds: 20261128 / 20261205 / 20261219','학습용 7-class 보조 head는 추론에서 제외'])
c+=node('blend021',1120,200,424,240,'최종 021 · 기존 018에 L을 혼합', ['021 = .9 × 018 + .1 × L(mean3)','각 입력은 이미 hardness 단위의 예측값','잔차는 018 내부에서 더한 뒤 .9를 곱함','추가 affine · 출력 clipping 없음','Public RMSE 11.4878517800 (저장 기록)'],True)
c+=txt(40,864,'추론: 저장된 파라미터만 사용 · D4 / TTA 없음 · test에서 fit / bootstrap / 선택 없음','body')
c+=txt(40,900,'학습과의 차이: train OOF로 보정·잔차 선택, backbone은 동결, 추론 시 dropout·보조 분류는 사용하지 않음','body')
make('overall-021','최고 모델 021 — 두 경로의 마지막 결합','코드·설정·저장 파라미터 대조 · 추론을 새로 실행하지 않고 확인한 구조',1040,c,'산호색: 018 내부 결합과 최종 021 결합 · 화살표: 추론 값의 흐름 · 회색: 입력·고정 연산·헤드')

c=''.join([arrow('M 520 280 H 600'),arrow('M 780 320 V 368'),arrow('M 960 440 H 1040'),arrow('M 520 620 H 552 Q 560 620 560 628 V 640 Q 560 648 568 648 H 600'),arrow('M 960 648 H 1112 Q 1120 648 1120 640 V 576'),arrow('M 520 848 H 600'),arrow('M 960 848 H 1040'),arrow('M 1280 576 V 760')])
c+=node('base4',40,192,480,480,'기본 4개 모델 · 각자 hardness로 출력', ['KRR: compact26 → MinMax clip → StandardScaler','block weight → RBF kernel (bandwidth 4, alpha .1)','CNN: BOX 128² /255 → train μ,σ normalization','Conv 1→16→32→64→96; stride2, GN4, SiLU','kernels 5/3/3/3 · padding 2/1/1/1','spatial: 128→64→32→16→8','mean96 + std96 + input mean,std2 = 194','194→64→1; SiLU, Dropout(.15); 3 seed 평균','DINO-S/B: BICUBIC 504² → RGB / ImageNet norm','동결 12 blocks · width S384 / B768','layers [5,8,11] × (CLS, mean, std)','S3456 / B6912 → StandardScaler → 64 → 1','GELU, Dropout(.25); S/B 각각 3 seed 평균','B6912 특징은 short-B로도 분기'])
c+=node('raw',600,192,360,128,'① 보정 전 기본 혼합', ['raw = .3 KRR + .3 CNNmean3','        + .2 Smean3 + .2 Bmean3'])
c+=node('cal',600,368,360,168,'② affine 보정', ['cal = target_center','      + slope × (raw − prediction_center)','slope = 1.110354875458264','intercept = −21.30912792768993'])
c+=node('short',600,584,360,128,'별도 short-B', ['B6912 → own scaler → 32 → 1','GELU · Dropout(.5) · seed 20270215'])
c+=node('anchor',1040,400,480,176,'③ short-B를 보정 후 혼합', ['base = .9 × cal + .1 × shortB','short-B에는 위 affine을 적용하지 않음','두 헤드는 각각 target scale을 복원','이 base에 잔차를 더해야 018이 됨'],True)
c+=node('features',40,720,480,224,'잔차 입력 · 585차원', ['physics57 = compact26 + denoised morphology31','B final patch tokens: 1296 × 768','fixed QR projection: 768 → 32; seed 20271201','36 × 36 × 32 → adjacent-difference Gram','32 × 32 대칭행렬의 상삼각 528차원','MinMax clip → StandardScaler → block weight'])
c+=node('residual',600,760,360,184,'저장된 잔차 모델 3개 블록', ['none / ridge1 / ridge.1 / rbf1 / rbf.1','블록별 5개 모델을 저장 가중치로 합침','가중치: 512 bootstrap 승자 빈도','3 honest blocks: n = 167 / 167 / 166','추론에는 bootstrap 재표집 없음'])
c+=node('018final',1040,760,480,184,'④ 잔차 블록 평균을 마지막에 더함', ['r_j = Σ_k w[j,k] × residual[j,k](X585)','018 = base + (r_0 + r_1 + r_2) / 3','잔차에 .1 또는 .9를 따로 곱하지 않음','그 다음 021에서 .9 × 018 + .1 × L','018 · 021 모두 output clipping 없음'],True)
c+=txt(40,1012,'학습: nested group OOF로 보정·잔차 평가 / bootstrap 가중치 결정 · 추론: 저장 scaler·모델·가중치만 적용','body')
c+=txt(40,1044,'CNN seeds 20261003/20261023/20261043 (마지막 5 epoch weight 평균) · S/B seeds 20261128/20261205/20261219','body')
make('detail-018','018 상세 — 보정, short-B, 잔차의 정확한 순서','같은 256² grayscale 이미지가 기본 모델과 잔차 특징 경로로 입력됩니다. ① → ② → ③ → ④ 순서가 핵심입니다.',1160,c,'산호색: base 결합과 잔차 합산 · 기본 모델 4개를 한 노드로 묶음 · nested 학습 절차는 하단 주석으로 축약')
print('created overall-021.html, detail-018.html')
