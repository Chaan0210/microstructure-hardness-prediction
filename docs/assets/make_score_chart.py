"""제출 이력 JSON에서 점수 추이 SVG와 docs/submissions.md를 만든다 (표준 라이브러리만 사용).

    python docs/assets/make_score_chart.py
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = _repo_path(ROOT, 'final/package/verification/submission-score-ranking.json')
SVG = _repo_path(ROOT, 'docs/assets/score-progression.svg')
TABLE = _repo_path(ROOT, 'docs/submissions.md')

# 제출 번호 -> 그 제출에서 바뀐 것 (기록 문서 기준)
NOTES = {
    1: '첫 제출',
    17: 'best021: 018 앙상블 0.9 + DINOv2-L MLP 0.1',
    19: 'Cellpose-SAM 결정립 측정 특징 + KRR, 021과 0.5 결합 (study 220)',
    20: '결정립 특징 v3 + TabPFN v2',
    21: 'TabPFN v2 seed 3개 + 잡음 이미지 4-view 분할 TTA',
    22: '열화 이미지로 미세조정한 Cellpose (push-300)',
    25: '라벨 없이 학습한 화소 U-Net 상(phase) 지도 (push-400)',
    28: 'TabPFN-3.5 x U-Net 변형 4개 평균',
    31: '블록 이질성 + U-Net darkness z (038)',
    34: '038·039·044 평균, 결합 비중 0.8',
    37: 'Ridge·logRidge 파트너 결합 (push-900)',
    40: '051 + GrainSet 0.05 + GMM 0.15 (053)',
    44: '최종 계보에서 제외',
    45: '최종 계보에서 제외',
    47: '원본 열화 사본으로 증강한 Ridge 파트너 (push-1200)',
    52: '저잡음 행은 이질성·밝기 열 제외 (073)',
    53: 'TabPFN 12개에 cumulant·contrast 특징 추가 (076)',
    56: '076 + 잡음 구간 잔차 CNN 보정 (082)',
    57: '082 + TabPFN 블록에 sigma 2·4 cumulant 열 추가 (083)',
    58: 'Cellpose flow·확신도 연속 특징 38개 (F38)',
    59: 'train 저잡음 250장으로 학습한 source encoder 2개',
    62: '**최종 선택**: 마지막 readout을 HGB 3개로 교체',
}
CHART_MARKS = {19: '결정립 측정', 25: '화소 U-Net', 31: '블록 이질성', 47: '열화 사본 증강', 58: 'flow 특징', 62: '최종 #62'}

rows = json.loads(SRC.read_text())['rows']
rows.sort(key=lambda r: r['number'])
assert [r['number'] for r in rows] == list(range(1, 63)), '제출 #1~#62가 중복·누락 없이 필요합니다.'
assert all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
           for r in rows for k in ('public', 'private')), 'Public·Private 점수가 모두 필요합니다.'
by_n = {r['number']: r for r in rows}

# ---------- SVG ----------
W, H = 880, 440
L, R, T, B = 60, 120, 64, 52
X0, X1, Y0, Y1 = 1, 62, 8.5, 13.25
px = lambda n: L + (n - X0) / (X1 - X0) * (W - L - R)
py = lambda v: T + (Y1 - v) / (Y1 - Y0) * (H - T - B)


def path(key):
    d, prev = [], None
    for r in rows:
        n = r['number']
        if r[key] is None:  # 점수가 없는 구간은 선을 끊는다
            continue
        cmd = 'L' if prev is not None and n == prev + 1 else 'M'
        d.append(f'{cmd}{px(n):.1f},{py(r[key]):.1f}')
        prev = n
    return ' '.join(d)


parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">제출별 RMSE 추이</title>
<desc id="d">제출 62건의 Public·Private RMSE. 첫 제출 Public 12.886, Private 13.030에서 최종 #62 Public 8.820, Private 9.697로 낮아졌다. 표는 docs/submissions.md에 있다.</desc>
<style>
  svg {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e6e5e0; --s1:#2a78d6; --s2:#eb6834; }}
  @media (prefers-color-scheme: dark) {{
    svg {{ --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#33332f; --s1:#3987e5; --s2:#d95926; }}
  }}
  text {{ font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", sans-serif; fill: var(--ink2); font-size: 12px; }}
  .title {{ fill: var(--ink); font-size: 15px; font-weight: 600; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .guide {{ stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 2 3; opacity: .55; }}
  .end {{ fill: var(--ink); font-weight: 600; }}
</style>
<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>
<text class="title" x="{L}" y="28">제출별 RMSE (낮을수록 좋음)</text>
<line x1="{L}" y1="46" x2="{L + 18}" y2="46" stroke="var(--s1)" stroke-width="2"/><text x="{L + 24}" y="50">Public (test 30%)</text>
<line x1="{L + 150}" y1="46" x2="{L + 168}" y2="46" stroke="var(--s2)" stroke-width="2"/><text x="{L + 174}" y="50">Private (test 70%)</text>''']

for v in range(9, 14):
    parts.append(f'<line class="grid" x1="{L}" y1="{py(v):.1f}" x2="{W - R}" y2="{py(v):.1f}"/>'
                 f'<text x="{L - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
for n in (1, 10, 20, 30, 40, 50, 62):
    parts.append(f'<text x="{px(n):.1f}" y="{H - B + 20}" text-anchor="middle">#{n}</text>')
parts.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 12}" text-anchor="middle">플랫폼 제출 번호</text>')

for i, (n, label) in enumerate(CHART_MARKS.items()):
    r = by_n[n]
    ly = T + 14 + (i % 2) * 16
    parts.append(f'<line class="guide" x1="{px(n):.1f}" y1="{ly + 4}" x2="{px(n):.1f}" y2="{py(r["private"]) - 6:.1f}"/>'
                 f'<text x="{px(n):.1f}" y="{ly}" text-anchor="middle">{label}</text>')

parts.append(f'<path d="{path("private")}" fill="none" stroke="var(--s2)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
parts.append(f'<path d="{path("public")}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
last = by_n[62]
for key, color in (('private', 's2'), ('public', 's1')):
    x, y = px(62), py(last[key])
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--{color})" stroke="var(--surface)" stroke-width="2"/>'
                 f'<text class="end" x="{x + 10:.1f}" y="{y + 4:.1f}">{key.capitalize()} {last[key]:.3f}</text>')
parts.append('</svg>')
SVG.write_text('\n'.join(parts) + '\n')

# ---------- 표 ----------
best = float('inf')
lines = ['# 제출 이력',
         '',
         '최종 수상 순위는 **3위**다. 리더보드 점수는 제출별 예측 성능의 기록이며 최종 평가 순위와 구분한다.',
         '',
         '플랫폼 제출 62건의 Public·Private RMSE다. 사용자 제공 전체 제출 이력과 파일명·점수를 대조했다(원본: '
         '`final/package/verification/submission-score-ranking.json`).',
         '',
         '- 플랫폼 제출 번호(#1–#62)와 후보 파일 번호(`submission-candidate-NNN.csv`)는 서로 다른 번호 체계다.',
         '- ▼는 그 시점까지의 Public 최고 갱신이다. Private 점수는 대회 종료 후 공개됐으므로 제출 당시에는 볼 수 없었다.',
         '',
         '![제출별 RMSE](assets/score-progression.svg)',
         '',
         '| # | 파일 | Public | Private | 변경 내용 |',
         '|---:|---|---:|---:|---|']
for r in rows:
    mark = ''
    if r['public'] < best:
        best, mark = r['public'], ' ▼'
    lines.append(f"| {r['number']} | `{r['filename']}` | {r['public']:.4f}{mark} | {'—' if r['private'] is None else f"{r['private']:.4f}"} | {NOTES.get(r['number'], '')} |")
TABLE.write_text('\n'.join(lines) + '\n')
print('wrote', SVG.relative_to(ROOT), TABLE.relative_to(ROOT))
