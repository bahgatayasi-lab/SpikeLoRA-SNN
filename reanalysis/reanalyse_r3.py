"""Reanalyse retained SpikeLoRA-TS R2 results. No training or model inference.

Usage: python reanalyse_r3.py --data-dir /path/to/original/csvs
The four original datasets are hash-checked; no model files are loaded. The
source-block sign-flip calculation is an exploratory reference calculation
under independent, sign-symmetric source blocks, not a population guarantee.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SOURCES = {'SR':'Palestine', 'WS':'Palestine', 'WP':'Turkey', 'EC':'Morocco'}
TASKS = ['SR','WS','WP','EC']
H = np.array([1,2,4,8,24], dtype=np.int64)
L = 96
SPECS = {
 'SR':('Palestine-Solar.csv','date','GHI',15,['Temperature','Cloud Type','Dew Point','Relative Humidity','Solar Zenith Angle','Pressure','Wind Direction','Wind Speed','GHI'], 'd30c392eddf014df2a416d4071fc87fbde68c18b8222041368028de94289252f',105001),
 'WS':('Palestine-wind.csv','date','Wind Speed',15,['Temperature','Cloud Type','Dew Point','Relative Humidity','Pressure','Wind Direction','Wind Speed'], '3f0adac44d33e1dccaba3b871e9d7b872ff9529151a4a3028d672307d75cedd0',105001),
 'WP':('Turky-Wind-power-Turbine.csv','date','LV ActivePower (kW)',10,['Wind Speed (m/s)','Wind Direction ('+chr(176)+')','LV ActivePower (kW)'], '768521f386057b400c9eb5e5d45c124a0c4b5129e0f13d8ff4350099d14e5870',48342),
 'EC':('Moroco-power-consumption.csv','Datetime','PowerConsumption_Zone1',10,['Temperature','Humidity','WindSpeed','PowerConsumption_Zone1'], '9c44b9af0cb1923a3fcbcef08826e13f8a405ebf68a8a072597b0feb17e45769',52297)
}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def split(anchors: np.ndarray, ratios: list, names: list) -> dict:
    bounds=[0]; cum=0.0
    for r in ratios[:-1]:
        cum += r; bounds.append(math.floor(len(anchors)*cum))
    bounds.append(len(anchors))
    raw=[anchors[bounds[i]:bounds[i+1]] for i in range(len(names))]
    return {name:(a[a+24<raw[i+1][0]] if i<len(raw)-1 else a)
            for i,(name,a) in enumerate(zip(names,raw))}

def tsummary(values: np.ndarray) -> dict:
    x=np.asarray(values,dtype=float); n=len(x)
    if n != 3 or not np.isfinite(x).all():
        raise ValueError('Exactly three finite paired seed values required')
    mean=float(x.mean()); sd=float(x.std(ddof=1))
    half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n))
    return {'n_seeds':n,'mean':mean,'sd':sd,'ci_low':mean-half,
            'ci_high':mean+half,'wins':int((x>0).sum())}

def write(frame: pd.DataFrame, out: Path, name: str) -> None:
    frame.to_csv(out/name,index=False,float_format='%.12g')

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--input-dir',type=Path,default=HERE/'inputs')
    ap.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=ap.parse_args(); out=args.out_dir; out.mkdir(parents=True,exist_ok=True)
    inp=args.input_dir
    e1=pd.read_csv(inp/'E1_R2_results.csv')
    e2=pd.read_csv(inp/'E2_R2_fewshot_results.csv')
    e4=pd.read_csv(inp/'E4_R2_ablation_results.csv')
    e3=pd.read_csv(inp/'E3_R2_pairwise_transfer_results.csv')
    ramp=pd.read_csv(inp/'E2_R2_ramp_regime_analysis.csv')
    assert len(e1)==84 and len(e2)==108 and len(e3)==54 and len(e4)==24
    assert not e1.duplicated(['task','model','seed']).any()
    assert not e2.duplicated(['task','mode','fraction','seed']).any()
    assert not e3.duplicated(['source_task','target_task','mode','fraction','seed']).any()
    assert not e4.duplicated(['task','ablation_label','fraction','seed']).any()
    assert set(e2['protocol_id']) == {'SpikeLoRA-TS-R2-corrected-v1'}
    assert all(np.isfinite(d['test_rmse']).all() for d in [e1,e2,e3,e4])
    # Equal weights within each seed/fraction/task, then equal source weights.
    p=e2.pivot(index=['task','fraction','seed'],columns='mode',values='test_rmse').reset_index()
    p['difference']=p.lora-p.spikelora
    p['relative_pct']=100*p.difference/p.lora
    p['log_ratio']=np.log(p.lora/p.spikelora)
    p['source']=p.task.map(SOURCES)
    write(p,out,'R3_E2_paired_effects.csv')
    c=e2.groupby(['task','fraction','mode']).test_rmse.mean().unstack().reset_index()
    c['difference']=c.lora-c.spikelora
    c['relative_pct']=100*c.difference/c.lora
    write(c,out,'R3_E2_condition_effects.csv')
    task=p.groupby('task').log_ratio.mean().reindex(TASKS)
    source=task.groupby(task.index.map(SOURCES)).mean()
    hierarchy=[]
    for name,value in task.items(): hierarchy.append({'level':'task','name':name,'mean_log_ratio':value,'geometric_reduction_pct':100*(-np.expm1(-value))})
    for name,value in source.items(): hierarchy.append({'level':'source','name':name,'mean_log_ratio':value,'geometric_reduction_pct':100*(-np.expm1(-value))})
    for name,value in [('task-balanced',task.mean()),('source-balanced',source.mean())]:
        hierarchy.append({'level':'overall','name':name,'mean_log_ratio':value,'geometric_reduction_pct':100*(-np.expm1(-value))})
    for name in source.index:
        value=source.drop(name).mean()
        hierarchy.append({'level':'leave-one-source-out','name':name,'mean_log_ratio':value,'geometric_reduction_pct':100*(-np.expm1(-value))})
    write(pd.DataFrame(hierarchy),out,'R3_hierarchical_effects.csv')
    permutations=[]
    for signs in itertools.product([-1,1],repeat=len(source)):
        permutations.append({'source_order':','.join(source.index),'signs':','.join(map(str,signs)),
                             'statistic':float(np.mean(source.values*np.array(signs)))})
    stat=float(source.mean())
    block_p=float(np.mean([abs(q['statistic'])>=abs(stat)-1e-14 for q in permutations]))
    write(pd.DataFrame(permutations),out,'R3_source_block_sign_flips.csv')
    statistical={'run_wins':int((p.difference>0).sum()),'condition_wins':int((c.difference>0).sum()),
       'run_sign_independence_reference_p':float(stats.binomtest(int((p.difference>0).sum()),len(p)).pvalue),
       'condition_sign_independence_reference_p':float(stats.binomtest(int((c.difference>0).sum()),len(c)).pvalue),
       'run_relative_wilcoxon_independence_reference_p':float(stats.wilcoxon(p.relative_pct).pvalue),
       'condition_relative_wilcoxon_independence_reference_p':float(stats.wilcoxon(c.relative_pct).pvalue),
       'source_block_two_sided_reference_p':block_p,
       'source_block_transformations':len(permutations),
       'source_balanced_geometric_reduction_pct':float(100*(-np.expm1(-stat))),
       'note':'Source-block symmetry/independence assumptions are unverified with three sources. No confirmatory population inference; no independent resampling of seeds or fractions.'}
    # Same-period references; not recovery from unadapted E2 checkpoints.
    baseline=e1.groupby(['task','model']).test_rmse.mean()
    dense_rows=[]
    for t in TASKS:
        d=baseline.loc[t].drop('snn_tcn'); best=d.idxmin(); rd=float(d.min())
        for _,row in c[c.task==t].iterrows():
            for mode in ['fullft','lora','spikelora']:
                value=float(row[mode]); dense_rows.append({'task':t,'fraction':row.fraction,'mode':mode,
                 'dense_reference':best,'dense_rmse':rd,'E1_snn_rmse':float(baseline.loc[(t,'snn_tcn')]),
                 'adapted_rmse':value,'adapted_to_dense_ratio':value/rd,
                 'excess_over_dense_pct':100*(value/rd-1),
                 'E1_snn_to_dense_ratio':float(baseline.loc[(t,'snn_tcn')])/rd})
    write(pd.DataFrame(dense_rows),out,'R3_dense_reference_context.csv')
    # All horizon coverage, width and recorded calibration quantiles, seed means/SD.
    ints=[]
    for (t,f,m),g in e2.groupby(['task','fraction','mode']):
        for h in H:
            ints.append({'task':t,'fraction':f,'mode':m,'horizon':int(h),'minutes':int(h)*SPECS[t][3],
              **{key+'_'+agg:float(getattr(g[f'{col}_h{h}'],agg)()) for key,col in [('coverage','picp95'),('width','mpiw95'),('q','conformal_q95'),('rmse','rmse')] for agg in ['mean','std']}})
    write(pd.DataFrame(ints),out,'R3_E2_horizon_intervals.csv')
    e1ints=[]
    for (t,m),g in e1.groupby(['task','model']):
        for h in H:
            e1ints.append({'task':t,'model':m,'horizon':int(h),'minutes':int(h)*SPECS[t][3],
              **{key+'_'+agg:float(getattr(g[f'{col}_h{h}'],agg)()) for key,col in [('coverage','picp95'),('width','mpiw95'),('q','conformal_q95'),('rmse','rmse')] for agg in ['mean','std']}})
    write(pd.DataFrame(e1ints),out,'R3_E1_horizon_intervals.csv')

    # Matched ablation, with small-sample seed-only t intervals.
    w=e4.pivot(index='seed',columns='ablation_label',values='test_rmse')
    differences=w['LoRA-r8']-w['SpikeLoRA-r8-v0.05']
    rel=100*differences/w['LoRA-r8']
    write(pd.DataFrame({'seed':w.index,'LoRA':w['LoRA-r8'],'SpikeLoRA':w['SpikeLoRA-r8-v0.05'],
                        'difference':differences,'relative_pct':rel}),out,'R3_E4_paired_seeds.csv')
    statistical['E4_raw_paired']=tsummary(differences)
    statistical['E4_relative_paired']=tsummary(rel)
    ab=e4.groupby('ablation_label').agg(rmse=('test_rmse','mean'),rmse_sd=('test_rmse','std'),
       mae=('test_mae','mean'),mae_sd=('test_mae','std'),input_zero_pct=('encoder_sparsity_pct','mean'),
       gate_zero_pct=('spikelora_sparsity_pct','mean'),trainable_pct=('trainable_ratio','mean'),
       time_sec=('train_time_sec','mean')).reset_index()
    ab['trainable_pct']*=100
    write(ab,out,'R3_E4_joint_accuracy_sparsity.csv')
    # Original SR/WS/EC regime membership is retained, not reconstructed from
    # rounded or inverse-transformed targets. WP archived rows are invalid strata.
    valid=ramp[ramp.task.isin(['SR','WS','EC'])].copy()
    regimes=[]; strataintervals=[]
    for (t,f,reg),g in valid.groupby(['task','fraction','regime']):
        assert g.regime_n.nunique()==1 and len(g)==9
        w=g.pivot(index='seed',columns='mode')
        result={'task':t,'fraction':f,'regime':reg,'n_origins':int(g.regime_n.iloc[0])}
        for h,key in [(0,'test_rmse')]+[(int(h),f'rmse_h{h}') for h in H]:
            s=tsummary(w[key]['lora']-w[key]['spikelora'])
            for k,v in s.items():result[f'h{h}_{k}']=v
        regimes.append(result)
        for mode,gg in g.groupby('mode'):
            strataintervals.append({'task':t,'fraction':f,'regime':reg,'mode':mode,
              'n_origins':int(g.regime_n.iloc[0]),'picp':float(gg.picp95.mean()),'mpiw':float(gg.mpiw95.mean())})
    write(pd.DataFrame(regimes),out,'R3_ramp_paired_seed_uncertainty.csv')
    write(pd.DataFrame(strataintervals),out,'R3_ramp_interval_diagnostics.csv')
    thresholds=valid[['task','regime','regime_n','ramp_q33','ramp_q67']].drop_duplicates()
    write(thresholds,out,'R3_archived_valid_ramp_groups.csv')
    # Optimization exposure reflects full, untruncated training runs in R2.
    exposure=e2[['task','fraction','mode','seed','selected_adaptation_windows','best_epoch','epochs_ran','best_val_loss']].copy()
    exposure['batches_per_epoch']=np.ceil(exposure.selected_adaptation_windows/128).astype(int)
    exposure['selected_epoch_updates']=exposure.best_epoch*exposure.batches_per_epoch
    exposure['total_updates']=exposure.epochs_ran*exposure.batches_per_epoch
    write(exposure,out,'R3_optimization_exposure.csv')
    # Read-only reconstruction of the archived preprocessing.
    calendar=[]; targets=[]; covars=[]; profiles=[]; checks={}
    for t,(name,time_col,target,freq,cols,expected_sha,expected_n) in SPECS.items():
        file=args.data_dir/name
        if sha(file)!=expected_sha: raise ValueError('Dataset hash mismatch: '+name)
        df=pd.read_csv(file)[[time_col]+cols].copy()
        df[time_col]=pd.to_datetime(df[time_col],errors='raise',dayfirst=False)
        assert not df[time_col].duplicated().any()
        for col in cols:df[col]=pd.to_numeric(df[col],errors='coerce')
        df=df.sort_values(time_col).set_index(time_col)
        time=pd.date_range(df.index.min(),df.index.max(),freq=f'{freq}min')
        df=df.reindex(time); y=df[target].to_numpy(dtype=np.float32)
        observed=df[target].notna().to_numpy(); inputs=df[cols].ffill(limit=2)
        iv=inputs.notna().all(axis=1).to_numpy()
        anchors=np.arange(L-1,len(time)-24,dtype=np.int64)
        prefix=np.r_[0,np.cumsum(~iv)]; ok=(prefix[anchors+1]-prefix[anchors-L+1])==0
        for h in H:ok &= observed[anchors+h] & np.isfinite(y[anchors+h])
        anchors=anchors[ok]; assert len(anchors)==expected_n
        a=split(anchors,[.45,.05,.20,.05,.05,.20],['base_train','base_val','adapt_pool','adapt_val','calib','test'])
        e=split(anchors,[.7,.05,.05,.2],['train','val','calib','test'])
        assert np.array_equal(a['test'],e['test']) and np.array_equal(a['calib'],e['calib'])
        # One target value per unique forecast origin; never five duplicated targets.
        base_vals=y[a['base_train']]; base_vals=base_vals[np.isfinite(base_vals)].astype(float)
        test_vals=y[a['test']]; test_vals=test_vals[np.isfinite(test_vals)].astype(float)
        blocks={f'adapt_{int(f*100)}':a['adapt_pool'][-math.ceil(len(a['adapt_pool'])*f):] for f in [.1,.2,.5]}
        blocks.update({k:a[k] for k in ['adapt_val','calib','test']})
        for b,ix in blocks.items():
            vals=y[ix]; vals=vals[np.isfinite(vals)].astype(float)
            calendar.append({'task':t,'block':b,'origins':len(ix),'first_origin':str(time[ix[0]]),
              'last_origin':str(time[ix[-1]]),'elapsed_days':(time[ix[-1]]-time[ix[0]]).total_seconds()/86400,
              'first_input':str(time[ix[0]-95]),'last_label':str(time[ix[-1]+24]),
              'gap_to_test_days':(time[a['test'][0]]-time[ix[-1]]).total_seconds()/86400,
              'batches_per_epoch':math.ceil(len(ix)/128)})
            targetrow={'task':t,'block':b,'finite_origins':len(vals),'nonfinite_origins':len(ix)-len(vals),
               'mean':float(vals.mean()),'sd':float(vals.std(ddof=0)),'median':float(np.median(vals)),
               'q10':float(np.quantile(vals,.1)),'q90':float(np.quantile(vals,.9)),
               'zero_pct':float(100*np.mean(vals==0)), 'negative_pct':float(100*np.mean(vals<0)),
               'normalized_W1_to_test':float(stats.wasserstein_distance(vals,test_vals)/base_vals.std(ddof=0))}
            if t=='SR':targetrow['daylight_pct']=float(100*np.mean(df.iloc[ix]['Solar Zenith Angle']<90))
            targets.append(targetrow)
            for col in cols:
                # Categorical and directional variables require other distances;
                # do not treat their numerical codes as Euclidean measurements.
                if col==target or 'Direction' in col or col=='Cloud Type':continue
                cv=df.iloc[ix][col].dropna().to_numpy(dtype=float)
                tv=df.iloc[a['test']][col].dropna().to_numpy(dtype=float)
                bv=df.iloc[a['base_train']][col].dropna().to_numpy(dtype=float)
                covars.append({'task':t,'block':b,'variable':col,'n':len(cv),'mean':float(cv.mean()),
                  'sd':float(cv.std(ddof=0)), 'q10':float(np.quantile(cv,.1)), 'q90':float(np.quantile(cv,.9)),
                  'normalized_W1_to_test':float(stats.wasserstein_distance(cv,tv)/max(bv.std(ddof=0),1e-12))})
            for feature,arr in [('month',time[ix].month),('dayofweek',time[ix].dayofweek),('hour',time[ix].hour)]:
                unique,counts=np.unique(arr,return_counts=True)
                for value,n in zip(unique,counts):profiles.append({'task':t,'block':b,'calendar_variable':feature,'value':int(value),'n':int(n),'proportion':float(n/len(ix))})
        ix=a['test']; magnitude=np.max(np.abs(y[ix[:,None]+H]-y[ix,None]),axis=1)
        finite=np.isfinite(magnitude); quant=np.quantile(magnitude[finite],[1/3,2/3])
        labels=np.where(magnitude[finite]<=quant[0],0,np.where(magnitude[finite]<=quant[1],1,2))
        check={'dataset_sha256':sha(file),'same_E1_E2_test':True,'same_E1_E2_calib':True,
          'n_valid_windows':len(anchors),'split_counts':{k:len(v) for k,v in a.items()},
          'finite_ramp_origins':int(finite.sum()),'nonfinite_ramp_origins':int((~finite).sum()),
          'nonfinite_origin_times':[str(v) for v in time[ix[~finite]]],
          'raw_target_finite_ramp_q33':float(quant[0]),'raw_target_finite_ramp_q67':float(quant[1]),
          'raw_target_finite_ramp_counts':[int((labels==j).sum()) for j in range(3)],
          'ramp_median':float(np.median(magnitude[finite])), 'ramp_q90':float(np.quantile(magnitude[finite],.9)),
          'ramp_zero_pct':float(100*np.mean(magnitude[finite]==0)),
          'warning':'Raw-data ramp reconstruction is not a replacement for archived inverse-transformed prediction-level regime membership, especially at ties.'}
        checks[t]=check
    write(pd.DataFrame(calendar),out,'R3_calendar_coverage.csv')
    write(pd.DataFrame(targets),out,'R3_target_distribution_audit.csv')
    write(pd.DataFrame(covars),out,'R3_covariate_distribution_audit.csv')
    write(pd.DataFrame(profiles),out,'R3_calendar_composition.csv')
    sr=pd.read_csv(args.data_dir/SPECS['SR'][0]); ws=pd.read_csv(args.data_dir/SPECS['WS'][0])
    shared={col:sr[col].equals(ws[col]) for col in set(sr.columns)&set(ws.columns)}
    assert all(shared.values())
    summary={'no_training':True,'no_model_inference':True,'statistics':statistical,
      'data_audit':checks,'SR_WS_shared_columns_identical':shared,
      'input_hashes':{p.name:sha(p) for p in inp.glob('*') if p.is_file()},
      'environment':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__}}
    (out/'R3_analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(statistical,indent=2))
    print(pd.DataFrame(targets)[['task','block','mean','sd','normalized_W1_to_test']].to_string(index=False))

if __name__=='__main__':main()
