"""
REAL end-to-end performance analysis on the extracted features.
Trains the attribute model, then measures on a held-out val set:
  - per-attribute accuracy (gates / graders / trivial)
  - per-organ accuracy
  - REAL Edge-F1   = model's predicted answers -> traversal engine -> vs GT edges
  - REAL report token-F1 = model's predicted answers -> compose_report -> vs GT report
These are the ACTUAL numbers (not the perfect-attribute ceiling).
"""
import json, os, random, sys, time
from collections import Counter, defaultdict
import torch
torch.set_num_threads(8)
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset, compute_class_weights
from src.model import REGAttributeModel, masked_multitask_loss, load_label_maps, default_criticality_weights
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot, edge_set

FEAT=os.environ.get("REG_FEAT_DIR", "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0")
LAB ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM  ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT ="/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
PER_ORGAN=int(sys.argv[1]) if len(sys.argv)>1 else 500   # slides/organ for the analysis subset
EPOCHS  =int(sys.argv[2]) if len(sys.argv)>2 else 8

lm=load_label_maps(LM)
ds=REGFeatureDataset(FEAT, LAB, lm)
print(f"total paired slides available: {len(ds)}", flush=True)

# balanced subset for a fast-but-representative analysis
by_org=defaultdict(list)
for i,s in enumerate(ds.samples): by_org[s["organ"]].append(i)
rng=random.Random(0); idxs=[]
for o,ii in by_org.items():
    rng.shuffle(ii); idxs+=ii[:PER_ORGAN]
rng.shuffle(idxs)
nval=int(0.18*len(idxs)); val,train=idxs[:nval], idxs[nval:]
print(f"analysis subset: {len(idxs)} (train {len(train)} / val {nval}) | {PER_ORGAN}/organ | {EPOCHS} epochs", flush=True)

device="cuda" if torch.cuda.is_available() else "cpu"; print(f"device: {device}", flush=True)
crit=default_criticality_weights(lm)
cw={q:w.to(device) for q,w in compute_class_weights(LAB, lm, list(lm.keys())).items()}
model=REGAttributeModel(lm, in_dim=1536, per_task_attention=False).to(device)
opt=torch.optim.AdamW(model.parameters(), lr=3e-4)

t0=time.time()
for ep in range(1,EPOCHS+1):
    model.train(); rng.shuffle(train); opt.zero_grad()
    for n,i in enumerate(train,1):
        f,t,o,s=ds[i]
        if not t: continue
        f=f.to(device)
        lg=model(f, questions=list(t.keys()))
        loss=masked_multitask_loss(lg,t,crit,cw)/8; loss.backward()
        if n%8==0: opt.step(); opt.zero_grad()
    opt.step()
    print(f"  epoch {ep}/{EPOCHS} done ({time.time()-t0:.0f}s)", flush=True)

# ===== EVAL =====
cot={c['id']:c for c in json.load(open(COT))}
graph,roots=load_graph()
def gt_edges(c): return {(s['question'],s['next_question']) for s in c['chain-of-thought'] if s['question']}
def gt_report(c):
    for s in c['chain-of-thought']:
        if s['question']=="What is the final pathology report?": return s['answer']
def tok_f1(a,b):
    A,B=(a or "").split(),(b or "").split()
    if not A and not B: return 1.0
    i=sum((Counter(A)&Counter(B)).values()); p=i/len(A) if A else 0; r=i/len(B) if B else 0
    return 2*p*r/(p+r) if p+r else 0.0
def e_f1(P,T):
    if not P and not T: return 1.0
    tp=len(P&T); pr=tp/len(P) if P else 0; rc=tp/len(T) if T else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

model.eval()
attr_cor=Counter(); attr_tot=Counter()
org_cor=Counter(); org_tot=Counter()
edge_by=defaultdict(list); rep_by=defaultdict(list)
with torch.no_grad():
    for i in val:
        f,t,o,sid=ds[i]
        if not t: continue
        f=f.to(device)
        # ONE forward over all asked questions -> predicted answers
        lg=model(f, questions=list(t.keys()))
        pred={q: lm[q][int(lg[q].argmax())] for q in t}
        for q,y in t.items():
            ok=int(lg[q].argmax())==y; attr_cor[q]+=ok; attr_tot[q]+=1
            org_cor[o]+=ok; org_tot[o]+=1
        # real Edge-F1 (engine driven by predicted answers) + report-F1
        c=cot.get(sid if sid.endswith('.tiff') else sid+'.tiff')
        if c:
            steps=assemble_cot(o, lambda q: pred.get(q), graph, roots, threshold=0.10)
            edge_by[o].append(e_f1(edge_set(steps), gt_edges(c)))
            rep_by[o].append(tok_f1(compose_report(pred), gt_report(c)))

GATES=["Is there any abnormality present?","Is there any neoplasm present?","What is the histologic type of neoplasm?"]
GRADERS=["Is there any invasion present?","What is the Gleason score?","What is the grade of neoplasm?",
         "What is the score for nuclear pleomorphism?","What is the score for mitotic rate?","What is the score for tubular differentiation?"]
print("\n================ REAL PERFORMANCE (val set, model-predicted answers) ================")
print("\n[GATES — drive Edge-F1]")
for q in GATES:
    if attr_tot[q]: print(f"  {attr_cor[q]/attr_tot[q]:.3f}  {q}  (n={attr_tot[q]})")
print("\n[GRADERS — drive MESS/report quality]")
for q in GRADERS:
    if attr_tot[q]: print(f"  {attr_cor[q]/attr_tot[q]:.3f}  {q}  (n={attr_tot[q]})")
print(f"\n[Overall attribute accuracy] {sum(attr_cor.values())/sum(attr_tot.values()):.3f}")
print("\n[Per-organ: attribute acc | REAL Edge-F1 | REAL report tokF1]")
print(f"  {'organ':10s}{'attr_acc':>9}{'edge_f1':>9}{'report':>9}")
ae=[];ee=[];re=[]
for o in sorted(org_tot):
    a=org_cor[o]/org_tot[o]; e=sum(edge_by[o])/len(edge_by[o]) if edge_by[o] else 0; r=sum(rep_by[o])/len(rep_by[o]) if rep_by[o] else 0
    ae.append(a);ee.append(e);re.append(r)
    print(f"  {o:10s}{a:>9.3f}{e:>9.3f}{r:>9.3f}")
print(f"  {'OVERALL':10s}{sum(ae)/len(ae):>9.3f}{sum(ee)/len(ee):>9.3f}{sum(re)/len(re):>9.3f}")
print("\n(Edge-F1 = Metric-A structure; report tokF1 ~ final-report component; both with REAL model predictions.)")

# save the trained model (usable for Step 3 scoring).
# OUT path is env-configurable so the otsu-retrain does NOT clobber the WORKING model checkpoint
# (reg_attr_analysis.pt). Default keeps the original name for backward compatibility.
ck="/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"; os.makedirs(ck, exist_ok=True)
out_ckpt=os.environ.get("REG_OUT_CKPT", ck+"/reg_attr_analysis.pt")
torch.save({"model":model.state_dict(),"label_maps":lm,"per_task_attention":False}, out_ckpt)
print("\nsaved checkpoint -> "+out_ckpt)
