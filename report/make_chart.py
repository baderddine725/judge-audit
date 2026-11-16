import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Inter","font.size":9,"svg.fonttype":"none"})
rows=list(csv.DictReader(open("runs/A/results.tsv"),delimiter="\t"))
n=list(range(1,len(rows)+1)); v=[float(r["val_bpb"]) for r in rows]; st=[r["status"] for r in rows]
kept=[];cur=None
for x,s in zip(v,st):
    if s=="keep": cur=x
    kept.append(cur)
ACC="#2F6FDE"; GREY="#A9A9A9"; INK="#1f2328"; Q="#6b7280"
fig,ax=plt.subplots(figsize=(7.2,3.3),dpi=150)
ax.axhline(v[0],ls=(0,(4,4)),lw=0.8,color=GREY)
ax.text(45.6,v[0],f"baseline {v[0]:.3f}",va="center",ha="left",color=Q,fontsize=8)
ax.step(n,kept,where="post",color=ACC,lw=1.8,zorder=2)
ax.scatter([a for a,s in zip(n,st) if s!="keep"],[b for b,s in zip(v,st) if s!="keep"],s=16,facecolors="none",edgecolors=GREY,lw=0.9,zorder=3,label="discarded (git reset)")
ax.scatter([a for a,s in zip(n,st) if s=="keep"],[b for b,s in zip(v,st) if s=="keep"],s=26,color=ACC,zorder=4,label="kept")
ax.annotate(f"#2 batch 2$^{{15}}$→2$^{{13}}$: −{v[0]-v[1]:.3f}",xy=(2,v[1]),xytext=(4.5,1.615),color=INK,fontsize=8.5,arrowprops=dict(arrowstyle="-",color=GREY,lw=0.8))
ax.text(45.6,kept[-1],f"kept {kept[-1]:.4f}",va="center",ha="left",color=INK,fontsize=8,fontweight="bold")
ax.set_xlim(0,45.5); ax.set_ylim(1.535,1.715)
ax.set_xticks([1,10,20,30,40,45]); ax.set_xlabel("experiment #",color=Q); ax.set_ylabel("val_bpb (lower is better)",color=Q)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#cfcfcf")
ax.tick_params(colors=Q); ax.grid(axis="y",color="#eeeeee",lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False,loc="upper right",fontsize=8,bbox_to_anchor=(0.97,0.93))
fig.tight_layout(); fig.savefig("report/chart.svg",bbox_inches="tight")
print(len(rows), sum(s=="keep" for s in st), v[0], kept[-1], min(v))
