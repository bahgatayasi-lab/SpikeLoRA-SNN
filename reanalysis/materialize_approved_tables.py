"""Restore two approved CSV tables locally, verifying their original SHA-256.

No network, credentials, datasets, model training or model inference. The calendar
CSV is losslessly gzip-compressed; E2 horizon summaries are recalculated only from
retained per-run metrics and must match the approved CSV bytes exactly.
"""
from pathlib import Path
import collections, csv, gzip, hashlib, io, statistics
ROOT = Path(__file__).resolve().parent
EXPECTED = {
 'R3_calendar_composition.csv':'b11f6327cf2ac24229d0b05cce72ae59f678864a4f2cf5d6fed9e017e6fd1bb9',
 'R3_E2_horizon_intervals.csv':'682ec35e3b3261e676ba1eeaeb26bc49420f747af6004f5ba828a3d8913ca034'}
def digest(data): return hashlib.sha256(data).hexdigest()
def store(name, data):
    if digest(data)!=EXPECTED[name]: raise ValueError('Approved-table hash mismatch: '+name)
    path=ROOT/'results'/name
    if path.exists() and path.read_bytes()!=data: raise ValueError('Refusing to replace different existing table: '+name)
    if not path.exists(): path.write_bytes(data)
def materialize():
    if not (ROOT/'results/R3_calendar_composition.csv').exists():
        packed=(ROOT/'retained_tables/R3_calendar_composition.csv.gz').read_bytes()
        store('R3_calendar_composition.csv',gzip.decompress(packed))
    if not (ROOT/'results/R3_E2_horizon_intervals.csv').exists():
        source=ROOT/'inputs/E2_R2_fewshot_results.csv'
        if digest(source.read_bytes())!='e0e985ce810cb91b9e03923934b746f45e8f034953fa055d95b2948662e33f9a':
            raise ValueError('Original E2 input hash mismatch')
        groups=collections.defaultdict(list)
        with source.open() as handle:
            for r in csv.DictReader(handle):groups[(r['task'],float(r['fraction']),r['mode'])].append(r)
        rows=[]
        for (t,f,m),g in sorted(groups.items()):
            if len(g)!=3:raise ValueError('Three original seeds required')
            for h in (1,2,4,8,24):
                r={'task':t,'fraction':format(f,'.12g'),'mode':m,'horizon':h,'minutes':h*(15 if t in ('SR','WS') else 10)}
                for key,col in [('coverage','picp95'),('width','mpiw95'),('q','conformal_q95'),('rmse','rmse')]:
                    x=[float(v[f'{col}_h{h}']) for v in g]
                    r[key+'_mean']=format(sum(x)/len(x),'.12g');r[key+'_std']=format(statistics.stdev(x),'.12g')
                rows.append(r)
        out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows);store('R3_E2_horizon_intervals.csv',out.getvalue().encode())
    for name,h in EXPECTED.items():
        if digest((ROOT/'results'/name).read_bytes())!=h:raise ValueError('Table verification failed: '+name)
if __name__=='__main__':
    materialize();print('Two approved tables available; exact SHA-256 verified. No model execution.')
