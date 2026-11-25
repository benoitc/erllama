import array
import json
import math
from pathlib import Path

out = {}
passed = True
for case in ('pre', 'one'):
    def read(name):
        a = array.array('f')
        a.frombytes(Path(name+'-'+case+'.f32').read_bytes())
        return a
    ref, a, b = [read(n) for n in ('reference','split','follower')]
    assert len(a)==len(b)==len(ref)
    for layer in range(3):
        ix = range(layer*len(a)//3,(layer+1)*len(a)//3)
        x,y,z = [[s[i] for i in ix] for s in (ref,a,b)]
        def compare(u,v):
            uu=sum(t*t for t in u);vv=sum(t*t for t in v)
            if not uu or not vv: return {'cosine':0,'relative_rms':None}
            return {'cosine':sum(p*q for p,q in zip(u,v))/math.sqrt(uu*vv),'relative_rms':math.sqrt(sum((p-q)**2 for p,q in zip(u,v))/uu)}
        r0,r1=compare(x,y),compare(x,z)
        passed &= all(s['cosine']>0.995 and s['relative_rms'] is not None and s['relative_rms']<0.10 for s in (r0,r1))
        out[f'{case}-{41+layer}']={'rank0':r0,'rank1':r1,'ranks_equal':y==z,'sum_vs_reference':compare(x,[p+q for p,q in zip(y,z)])}
print(json.dumps(out, indent=2))
Path('features.json').write_text(json.dumps(out,indent=2)+'\n')
raise SystemExit(0 if passed else 1)
