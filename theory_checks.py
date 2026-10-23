"""Additional verification for the revised manuscript (seed 20261023)."""
import numpy as np, json
from scipy.special import expit, logit
rng=np.random.default_rng(20261023)
OUT={}
# ---- exact zero count for linear-rule pair (vectorized)
def nzeros(gA,pA,a,gB,pB,b):
    kA=gA-pA; kB=gB-pB; k0=gA-gB; kinf=pA-pB
    ratio=(b*kB)/(a*kA)
    with np.errstate(all='ignore'):
        ss=np.log(ratio)/(b-a)
    has=(ratio>0)&np.isfinite(ss)&(ss>0)
    kst=np.where(has,(pA-pB)+kA*np.exp(-a*np.where(has,ss,0))-kB*np.exp(-b*np.where(has,ss,0)),np.nan)
    z=np.where(has,(np.sign(k0)!=np.sign(kst)).astype(int)+(np.sign(kst)!=np.sign(kinf)).astype(int),(np.sign(k0)!=np.sign(kinf)).astype(int))
    return z
# sanity vs grid on known example
print("tight example zeros:",nzeros(np.array([.95]),np.array([.30]),np.array([5.]),np.array([.90]),np.array([.20]),np.array([.5])))
# ---- n=3 sharpness search: all three pairs with 2 exchanges
best=None; found=0; N=4_000_000; B=500_000
for _ in range(N//B):
    g=rng.uniform(0,1,(3,B)); p=rng.uniform(0,1,(3,B)); d=np.exp(rng.uniform(np.log(1e-2),np.log(1e2),(3,B)))
    z01=nzeros(g[0],p[0],d[0],g[1],p[1],d[1]); z02=nzeros(g[0],p[0],d[0],g[2],p[2],d[2]); z12=nzeros(g[1],p[1],d[1],g[2],p[2],d[2])
    tot=z01+z02+z12; i=np.argmax(tot)
    if best is None or tot[i]>best[0]: best=(int(tot[i]),[[float(g[k,i]),float(p[k,i]),float(d[k,i])] for k in range(3)])
    found+=int((tot==6).sum())
OUT["n3_search"]={"triples":N,"max_total_exchanges":best[0],"count_total6":found,"example":best[1]}
print(OUT["n3_search"])
# ---- opposite-direction check and near-tie diagnostics for mixed rule (lin vs log)
s=np.concatenate([np.geomspace(1e-6,1,3000),np.linspace(1,300,3000)[1:]])
lin=lambda g,p,d,t:p+(g-p)*np.exp(-d*t); lgt=lambda g,p,d,t:expit(logit(p)+(logit(g)-logit(p))*np.exp(-d*t))
def cnt(D): sg=np.sign(D); return np.sum(sg[:,1:]*sg[:,:-1]<0,axis=1)
opp_max=0; same3=[]; Nm=100000; Bm=5000; hist=np.zeros(5,int)
for _ in range(Nm//Bm):
    g1,p1,g2,p2=rng.uniform(1e-3,1-1e-3,(4,Bm,1)); dA,dB=np.exp(rng.uniform(np.log(1e-2),np.log(1e2),(2,Bm,1)))
    c=cnt(lin(g1,p1,dA,s)-lgt(g2,p2,dB,s)); hist+=np.bincount(c,minlength=5)[:5]
    opp=(np.sign(g1-p1)!=np.sign(g2-p2))[:,0]
    if opp.any(): opp_max=max(opp_max,int(c[opp].max()))
    for i in np.where(c>=3)[0]: same3.append([float(abs(g1[i,0]-g2[i,0])),float(abs(p1[i,0]-p2[i,0]))])
OUT["mixed_rule"]={"pairs":Nm,"hist_0to4":hist.tolist(),"max_exchanges_opposite_direction":opp_max,"near_tie_|dg|,|dpi|_for_>=3":same3}
print(OUT["mixed_rule"])
# ---- same-rule frequency (exact, 1e6 pairs)
g=rng.uniform(0,1,(2,1_000_000)); p=rng.uniform(0,1,(2,1_000_000)); d=np.exp(rng.uniform(np.log(1e-2),np.log(1e2),(2,1_000_000)))
z=nzeros(g[0],p[0],d[0],g[1],p[1],d[1]); OUT["same_rule_linear_exact"]={"pairs":10**6,"hist_0to2":np.bincount(z,minlength=3).tolist()}
print(OUT["same_rule_linear_exact"])
json.dump(OUT,open("theory_checks.json","w"),indent=1)
