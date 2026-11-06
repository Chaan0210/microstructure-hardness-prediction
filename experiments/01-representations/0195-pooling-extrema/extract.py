import common as c
from common import np,R,P
c.check();d=c.data();src=c.load(P/'artifacts/dinob-062/development_features.npz');bench=c.load(P/'artifacts/dinob-062/benchmark_features.npz');mom=c.load(P/'artifacts/patch-attention-092/token_moments.npz');meta=c.read(P/'artifacts/patch-attention-092/cache.json')
for x in [src,mom]:np.testing.assert_array_equal(x['ids'],d['ids'])
assert meta['ids']==d['ids'].tolist() and meta['all400_pooled_descriptor_exact'];np.testing.assert_array_equal(src['X'][:3],bench['X']);np.testing.assert_array_equal(d['ids'][:3],bench['ids'])
assert src['X'].shape==(400,6912) and np.isfinite(src['X']).all();out={n:np.empty((400,768),np.float32) for n in ['maximum','minimum','second_maximum','second_minimum']};sdev=np.empty((400,768));meanerr=[];stderr=[]
with c.Monitor('extraction'):
 for i,a in c.tokens():
  aa=a.astype(float);mean=aa.mean(0);std=aa.std(0);sdev[i]=std;mm=np.stack((mean,np.square(aa).mean(0)));np.testing.assert_array_equal(mm,mom['moments'][i]);np.testing.assert_allclose(mean,src['X'][i,5376:6144],atol=2e-5,rtol=2e-5);np.testing.assert_allclose(std,src['X'][i,6144:6912],atol=2e-5,rtol=2e-5)
  meanerr.append(float(abs(mean-src['X'][i,5376:6144]).max()));stderr.append(float(abs(std-src['X'][i,6144:6912]).max()));out['maximum'][i]=a.max(0);out['minimum'][i]=a.min(0);part=np.partition(a,[1,1294],axis=0);out['second_minimum'][i]=part[1];out['second_maximum'][i]=part[1294]
  if (i+1)%80==0:c.emit({'token_rows_completed':i+1,'total':400})
 c.arrays(R/'features.npz',ids=d['ids'],baseline=src['X'],final_std64=sdev,**out)
 gaps={}
 for arm,key,second in [('max','maximum','second_maximum'),('min','minimum','second_minimum')]:
  delta=abs(out[key].astype(float)-out[second].astype(float));relative=delta/np.maximum(sdev,1e-12);maxgap=relative.max(1);top=np.argsort(-maxgap,kind='stable')[:5];gaps[arm]={'raw_range':[float(out[key].min()),float(out[key].max())],'absolute_gap_quantiles':np.quantile(delta,[0,.5,.95,.99,1]).tolist(),'gap_over_population_std_quantiles':np.quantile(relative,[0,.5,.95,.99,1]).tolist(),'largest_relative_gap_images':[{'ID':str(d['ids'][i]),'max_channel_gap_over_std':float(maxgap[i])} for i in top]}
 r={'all400_moment_cache_exact':True,'all_baseline_CLS_mean_std_copied_exact':True,'benchmark_first3_exact':True,'final_mean_max_abs_legacy_difference':max(meanerr),'final_std_max_abs_legacy_difference':max(stderr),'CLS_verification_limit':c.plan['features']['consistency'],'feature_dimensions':{'baseline':6912,'plus_max':7680,'plus_min':7680},'raw_feature_dtype':'float32','Ridge_input_dtype':'float64','token_chunk_images':1,'extrema_diagnostics':gaps,'features_SHA256':c.sha(R/'features.npz'),'no_label_based_feature_fit':True};c.write(R/'extraction.json',r);c.emit(r)
c.check()
