import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import R,P,np
c.check();v=c.read(R/'validation.json');assert v['passed'] and v['OOF_SHA256']==c.sha(R/'OOF.npz');z=c.load(R/'OOF.npz');d=c.data()
for a,b in [('ids','ids'),('y','y'),('fold','fold'),('groups','g'),('best021','blend_L_MLP')]:np.testing.assert_array_equal(z[a],d[b])
y=z['y'];fold=z['fold'];ref=z['best021'];rmse=lambda p:float(np.sqrt(np.mean((y-p)**2)));Q={a:.9*ref+.1*z[a] for a in c.plan['arms']};scores={'best021':rmse(ref)}
for a in c.plan['arms']:scores[a]={'standalone':rmse(z[a]),'blend10':rmse(Q[a]),'fold_standalone':[float(np.sqrt(np.mean((y[fold==k]-z[a][fold==k])**2))) for k in range(5)],'fold_blend':[float(np.sqrt(np.mean((y[fold==k]-Q[a][fold==k])**2))) for k in range(5)],'residual_correlation_vs021':float(np.corrcoef(y-z[a],y-ref)[0,1])}
fg=[float(np.sqrt(np.mean((y[fold==k]-ref[fold==k])**2)))-scores['micronet']['fold_blend'][k] for k in range(5)];pair=[a-b for a,b in zip(scores['imagenet']['fold_blend'],scores['micronet']['fold_blend'])]
lossref=(y-ref)**2;lossB=(y-Q['micronet'])**2;lossA=(y-Q['imagenet'])**2;benefit=lossref-lossB;keep=np.argsort(benefit,kind='stable')[:-5];top5gain=float(np.sqrt(lossref[keep].mean())-np.sqrt(lossB[keep].mean()))
rng=np.random.default_rng(202610205);ci_vals=[];ab_vals=[]
for _ in range(10000):
 ix=rng.integers(400,size=400);ci_vals.append(float(np.sqrt(lossref[ix].mean())-np.sqrt(lossB[ix].mean())));ab_vals.append(float(np.sqrt(lossA[ix].mean())-np.sqrt(lossB[ix].mean())))
ci=np.quantile(ci_vals,[.025,.975]).tolist();gain=scores['best021']-scores['micronet']['blend10']
gate={'technical_validation':True,'standalone_B_better_A':scores['micronet']['standalone']<scores['imagenet']['standalone'],'blend_B_better_A':scores['micronet']['blend10']<scores['imagenet']['blend10'],'paired_blend_atleast3folds':sum(x>0 for x in pair)>=3,'gain_vs021_atleast_0_02':gain>=.02,'vs021_atleast3folds':sum(x>0 for x in fg)>=3,'CI_lower_positive':ci[0]>0,'top5_removal_positive':top5gain>0};advance=all(gate.values())
# Independent loop/count weighted check using the same RNG draws, without using vector RMSE code.
rng2=np.random.default_rng(202610205);ind=[]
for _ in range(10000):
 counts=np.bincount(rng2.integers(400,size=400),minlength=400);ind.append(float(np.sqrt(counts@lossref/400)-np.sqrt(counts@lossB/400)))
err=float(abs(np.array(ind)-ci_vals).max());assert err<1e-10
for name,p in [('imagenet',z['imagenet']),('micronet',z['micronet'])]:assert abs((sum((float(t)-float(a))**2 for t,a in zip(y,p))/400)**.5-scores[name]['standalone'])<1e-10
summary={'study':205,'status':'완료','scores':scores,'MicroNet_gain_vs_ImageNet_standalone':scores['imagenet']['standalone']-scores['micronet']['standalone'],'MicroNet_gain_vs_ImageNet_blend':scores['imagenet']['blend10']-scores['micronet']['blend10'],'gain_vs021':gain,'fold_gains_vs021':fg,'paired_blend_fold_gains_vs_ImageNet':pair,'conditional_CI95_gain_vs021':ci,'conditional_CI95_blend_gain_vs_ImageNet':np.quantile(ab_vals,[.025,.975]).tolist(),'top5_removed_gain':top5gain,'top5_IDs':[str(x) for x in z['ids'][np.argsort(benefit,kind='stable')[-5:]]],'positive_ID_count':int((benefit>0).sum()),'criteria':gate,'advance':advance,'decision':'고정 조건에서 후속 검토 기준 통과. 재분할·제출은 별도 범위.' if advance else '사전 기준 미달. 이 MicroNet mean2048 EDF32 분기를 종료하고 best021을 유지함.','limitations':['같은 개발400의 반복 사용이며 독립 holdout 아님.','조건부 CI는 누적 선택과 재학습 불확실성을 보정하지 않음.','가중치 출처의 효과이며 사전학습 데이터만의 인과 효과 아님.','HF 체크포인트의 v1.0/v1.1 대응은 미확인.','두 모델 모두 동결 mean2048+Ridge EDF32 조건; 모든 MicroNet 활용법의 성능 한계를 뜻하지 않음.'],'validation':{'passed':True,'main_Ridge_fits':10,'poison_fits':10,'permuted_y_fits':10,'independent_solver_fits':10,'gradient_fits':0,'independent_bootstrap_max_error':err,'max_independent_Ridge_prediction_error':v['max_independent_prediction_error']},'downloads':1,'installations':0,'holdout_test_full500_newCSV_upload':False,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'cost':0,'plan_SHA256':c.sha(R/'plan.json'),'OOF_SHA256':c.sha(R/'OOF.npz')}
c.check();c.write(R/'summary.json',summary);c.emit(summary)
