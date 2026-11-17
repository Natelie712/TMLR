"""Auditable paper figures. Presentation only: no fitting, selection or inference."""
import string
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from artifact_common import *

SHORT={'qwen3':'Qwen3-1.7B','llama3b':'Llama-3.2-3B','llama2':'Llama-2-7B'}
BLUE='#0072B2'
WC={0.:'#E69F00',.001:'#56B4E9',.005:'#009E73',.01:'#CC79A7',.05:'#D55E00',.1:'#8C564B',.25:'#6A3D9A',.5:'#B2182B',1.:'#555555'}
P=PAPER/'data/geometric_diagnostics'

def style():
    plt=plot_style()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
                         'axes.formatter.useoffset':False,'figure.constrained_layout.use':True})
    return plt

def lam(w):return r'$\lambda='+f'{w:g}'+r'$'
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)
def weights(m,c):return sorted({0.,.01,1.,selected(m)[c]})
def ecdf(ax,x,**kw):
    x=np.sort(np.asarray(x,dtype=float));assert len(x) and np.isfinite(x).all()
    ax.step(np.r_[x[0],x],np.r_[0.,np.arange(1,len(x)+1)/len(x)],where='post',**kw)
def decorate(ax):
    ax.grid(axis='y',color='#dddddd',lw=.5,alpha=.65);ax.set_axisbelow(True)
def badge(ax,i):ax.text(-.13,1.04,'('+string.ascii_lowercase[i]+')',transform=ax.transAxes,fontweight='bold',fontsize=10)
def save(fig,number,draft=False):
    folder=PAPER/'figures'/('drafts' if draft else '')
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        path=folder/f'figure_{number}.{ext}';tmp=path.with_suffix('.'+ext+'.tmp')
        fig.savefig(tmp,format=ext,dpi=180,bbox_inches='tight');tmp.replace(path)
        other=PAPER/'figures'/('' if draft else 'drafts')/path.name
        if other.exists():other.unlink()  # Superseded figure only; inputs stay untouched.
    import matplotlib.pyplot as plt
    plt.close(fig)

def main_results_figure():
    plt=style();source=rows(PAPER/'data/training_validation_test.csv');main=rows(PAPER/'data/main_results.csv')
    fig,axes=plt.subplots(2,3,figsize=(11,7.4),sharey='row')
    all_ci=[float(r[k]) for r in source if r['partition']=='benchmark' and r['method'] in ['Linear',f"lambda_{selected(r['model'])[r['concept']]}" ] for k in ['ci_low','ci_high']]
    top_lim=(5*np.floor(min(all_ci)/5)-3,5*np.ceil(max(all_ci)/5)+3)
    bot_lim=(5*np.floor(min(float(r['difference_ci_low']) for r in main)/5)-2,5*np.ceil(max(float(r['difference_ci_high']) for r in main)/5)+2)
    for j,m in enumerate(MODELS):
        ax=axes[0,j];bx=axes[1,j]
        for i,c in enumerate(NAMES):
            for off,method,color,marker in [(-.15,'Linear',BLUE,'o'),(.15,f'lambda_{selected(m)[c]}','#333333','s')]:
                r=next(r for r in source if r['model']==m and r['concept']==c and r['partition']=='benchmark' and r['method']==method)
                v,lo,hi=[float(r[k]) for k in ['positive_delta_pp','ci_low','ci_high']]
                ax.errorbar(v,i+off,xerr=[[v-lo],[hi-v]],fmt=marker,color=color,capsize=2,ms=4,lw=1)
            r=next(r for r in main if r['model']==m and r['concept']==c);v=float(r['difference_pp'])
            bx.errorbar(v,i,xerr=[[v-float(r['difference_ci_low'])],[float(r['difference_ci_high'])-v]],fmt='o',color='#333333',capsize=2,ms=4)
        for a,lim in [(ax,top_lim),(bx,bot_lim)]:
            a.axvline(0,color='#777777',ls=':',lw=.8);a.set_yticks(range(7),list(NAMES.values()));a.set_ylim(6.6,-.6);a.set_xlim(*lim);decorate(a)
        ax.set_title(SHORT[m]);ax.set_xlabel(r'Positive $\Delta P$ (pp)');bx.set_xlabel('Curveball − Linear (pp)')
        badge(ax,j);badge(bx,j+3)
    axes[0,0].legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='CAA Linear'),Line2D([],[],color='#333333',marker='s',ls='',label='Selected Curveball')],loc='lower right',fontsize=7)
    fig.suptitle('Official Test · exploratory fixed benchmark · pointwise paired 95% CIs',fontsize=11)
    save(fig,1)

def geometry_figures():
    plt=style();main_results_figure()
    audit=rows(P/'cluster_target_label_audit.csv');fig,axes=plt.subplots(2,2,figsize=(8.4,6.7))
    for i,c in enumerate(['hallucination','survival-instinct']):
        d=np.load(P/f'local_qwen3_{c}.npz');u=unit(d['directions']);v=unit(d['global_direction']);pca=PCA(2,random_state=42);xy=pca.fit_transform(np.vstack([u,v]))
        aa=sorted([r for r in audit if r['model']=='qwen3' and r['concept']==c],key=lambda r:int(r['cluster']))
        fractions=[json.loads(r['target_label_counts']).get('(A)',0)/int(r['question_records']) for r in aa]
        ax=axes[i,0];sc=ax.scatter(xy[:-1,0],xy[:-1,1],c=fractions,vmin=0,vmax=1,cmap='coolwarm',s=60,edgecolors='#444444',lw=.5)
        for k,(x,y) in enumerate(xy[:-1]):
            offset={2:(-10,9),6:(10,-12),3:(7,6),4:(10,14)}.get(k,(5,4)) if c=='survival-instinct' else (5,4)
            ax.annotate(str(k+1),(x,y),xytext=offset,textcoords='offset points',fontsize=7,
                        arrowprops=dict(arrowstyle='-',color='#888888',lw=.4) if offset!=(5,4) else None)
        ax.scatter(*xy[-1],marker='*',s=110,color='k',label='Global Linear');ax.set(xlabel='Unit-direction PC1',ylabel='Unit-direction PC2',title=NAMES[c]+f'\n2D PCA variance: {pca.explained_variance_ratio_.sum():.1%}');ax.legend(loc='lower right',fontsize=7)
        ax=axes[i,1];ecdf(ax,d['random_cosines'].ravel(),color='#888888',label='Random partitions (500)')
        ecdf(ax,d['cosines'],color='#D55E00',marker='o',ms=3,label='Activation clusters (8)')
        ax.set(xlim=(-1.03,1.03),ylim=(-.02,1.02),xlabel='Cosine with global CAA Linear',ylabel='Empirical CDF');ax.legend(loc='upper left',fontsize=7);decorate(ax)
    fig.colorbar(sc,ax=list(axes[:,0]),label='Fraction with target label A',shrink=.65,pad=.02)
    for i,ax in enumerate(axes.flat):badge(ax,i)
    fig.suptitle('Qwen3 · fitting pairs · cluster labels 1–8\nColor reveals answer-label composition; clustering is not proof of behavioral geometry.',fontsize=10)
    save(fig,4)

    d=np.load(P/'updates_qwen3_hallucination.npz');fig,axes=plt.subplots(1,3,figsize=(9,3.6))
    for i,(ax,w) in enumerate(zip(axes,[0.,.01,1.])):
        xy=d[f'projection_{w}'];ax.scatter(xy[:,0],xy[:,1],s=12,alpha=.55,color=WC[w]);unit_axes(ax)
        ax.set_title(lam(w)+(' · selected' if w==.01 else ''));badge(ax,i)
    fig.suptitle('Qwen3 Hallucination · validation completed-answer activations\nCommon orthonormal axes; no per-panel magnification',fontsize=10)
    save(fig,5)

    mcq=rows(P/'mcq_displacements.csv');fig,axes=plt.subplots(1,3,figsize=(10.5,4.5));ratios=[]
    cases=[('qwen3','hallucination'),('llama3b','hallucination'),('llama2','myopic-reward')]
    plotted=[]
    for j,(ax,(m,c)) in enumerate(zip(axes,cases)):
        rr=[r for r in mcq if r['model']==m and r['concept']==c];linear={r['index']:float(r['displacement']) for r in rr if r['method']=='Linear'}
        ax.axvline(1,color=BLUE,ls='--',lw=1,label=f'Linear RMS {np.sqrt(np.mean(np.array(list(linear.values()))**2)):.3f}')
        for w in sorted({0.,selected(m)[c],1.}):
            ss=[r for r in rr if r['method']==f'lambda_{w}'];a=np.array([float(r['displacement']) for r in ss]);r=a/np.array([linear[x['index']] for x in ss]);ratios.extend(r)
            ecdf(ax,r,color=WC[w],lw=1.4,label=lam(w)+f' · RMS {np.sqrt(np.mean(a*a)):.3f}')
            plotted.extend(dict(model=m,concept=c,index=x['index'],weight=w,actual_l2=float(v),ratio_to_linear=float(q)) for x,v,q in zip(ss,a,r))
        ax.set(title=SHORT[m]+'\n'+NAMES[c],xlabel=r'$\|\Delta h\|_2$ / actual Linear norm',ylabel='Empirical CDF',ylim=(-.02,1.02));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),fontsize=7);decorate(ax);badge(ax,j)
    for ax in axes:ax.set_xlim(0,max(1.1,max(ratios)*1.05))
    fig.suptitle('Validation MCQ decision token · natural displacement\nAxis normalization is for display only; legend RMS values are unscaled L2 norms.',fontsize=10)
    write_csv(PAPER/'data/figure_6_plot_data.csv',plotted);save(fig,6)

    eig=rows(P/'kernel_variance_reconstruction.csv');fig,axes=plt.subplots(2,3,figsize=(10.5,6.5),sharex=True,sharey='row')
    behavior_colors=plt.get_cmap('tab10').colors
    for j,m in enumerate(MODELS):
        for i,c in enumerate(NAMES):
            ss=sorted([r for r in eig if r['model']==m and r['concept']==c],key=lambda r:float(r['weight']));x=[float(r['weight']) for r in ss]
            axes[0,j].plot(x,[float(r['fit_relative_reconstruction']) for r in ss],'-o',color=behavior_colors[i],ms=3,label=NAMES[c])
            axes[1,j].plot(x,[float(r['retained_kernel_variance_fraction']) for r in ss],'-o',color=behavior_colors[i],ms=3)
        axes[0,j].set(title=SHORT[m],ylim=(0,.42),ylabel=r'Reconstruction $\|\hat X-X\|_F/\|X\|_F$')
        axes[1,j].axhline(.95,color='k',ls=':',lw=.8);axes[1,j].set(ylim=(0,1),xlabel=r'Quadratic weight $\lambda$',ylabel='Retained kernel variance');axes[1,j].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes[:,j]:
            ax.set_xscale('symlog',linthresh=.001);ax.set_xticks([0,.001,.01,.1,1],['0','0.001','0.01','0.1','1']);decorate(ax)
    for i,ax in enumerate(axes.flat):badge(ax,i)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Fitting data · 20 KPCA components · common vertical scales\nDotted line: 95% reference, not a criterion satisfied by these fits.',fontsize=10)
    save(fig,7)

    allw=sorted({w for m in MODELS for c in NAMES for w in weights(m,c)})
    fig,axes=plt.subplots(7,3,figsize=(9,12),sharex=True,sharey=True)
    for i,c in enumerate(NAMES):
        for j,m in enumerate(MODELS):
            ax=axes[i,j];d=np.load(P/f'updates_{m}_{c}.npz')
            for w in weights(m,c):ecdf(ax,d[f'projection_{w}'][:,0],color=WC[w],lw=1.2)
            ax.set(xlim=(-1.02,1.02),ylim=(-.02,1.02));decorate(ax)
            if i==0:ax.set_title(SHORT[m])
            if j==0:ax.set_ylabel(NAMES[c]+'\nEmpirical CDF',fontsize=8)
            if i==6:ax.set_xlabel('Update cosine with CAA Linear')
            ax.text(.02,.94,lam(selected(m)[c])+' selected',transform=ax.transAxes,va='top',fontsize=7)
            if min(d[f'projection_{w}'][:,0].min() for w in weights(m,c))>.995:
                ax.text(.06,.35,'All curves near cosine = 1',transform=ax.transAxes,fontsize=6.5,color='#666666')
    fig.legend([Line2D([],[],color=WC[w],lw=1.5) for w in allw],[lam(w) for w in allw],loc='outside lower center',ncol=len(allw),fontsize=8)
    fig.suptitle('Validation completed-answer activations · direction distributions\nEmpirical CDFs preserve point masses; no histogram density spikes.',fontsize=10)
    save(fig,12)
    for number,c in zip(range(13,20),['corrigible-neutral-HHH','coordinate-other-ais','hallucination','myopic-reward','survival-instinct','sycophancy','refusal']):
        fig,axes=plt.subplots(3,3,figsize=(10,8.4));maxratio=1.
        for i,m in enumerate(MODELS):
            d=np.load(P/f'updates_{m}_{c}.npz');vnorm=np.linalg.norm(d['global_linear'])
            for w in weights(m,c):
                xy=d[f'projection_{w}'];ratio=np.linalg.norm(d[f'update_{w}'],axis=1)/vnorm;maxratio=max(maxratio,float(ratio.max()))
                ecdf(axes[i,0],xy[:,0],color=WC[w],lw=1.2)
                axes[i,1].scatter(xy[:,0],xy[:,1],s=6,alpha=.45,color=WC[w])
                ecdf(axes[i,2],ratio,color=WC[w],lw=1.2,label=lam(w)+(' (selected)' if w==selected(m)[c] else ''))
            axes[i,0].set(title=SHORT[m],xlim=(-1.02,1.02),ylim=(-.02,1.02),xlabel='Cosine with CAA Linear',ylabel='Empirical CDF')
            unit_axes(axes[i,1]);axes[i,2].axvline(1,color=BLUE,ls='--',lw=.8)
            axes[i,2].set(xlabel=r'$\|\Delta h\|_2/\|v_{\rm Linear}\|_2$',ylabel='Empirical CDF',ylim=(-.02,1.02));axes[i,2].legend(fontsize=6.5,loc='upper left',bbox_to_anchor=(1.01,1))
            if min(d[f'projection_{w}'][:,0].min() for w in weights(m,c))>.995:
                axes[i,0].text(.05,.3,'All curves near cosine = 1',transform=axes[i,0].transAxes,fontsize=6.5,color='#666666')
            decorate(axes[i,0]);decorate(axes[i,2])
        for ax in axes[:,2]:ax.set_xlim(0,maxratio*1.05)
        for i,ax in enumerate(axes.flat):badge(ax,i)
        fig.suptitle(NAMES[c]+' · validation completed-answer activations\nShared unit-direction axes; magnitude normalized for display only.',fontsize=10)
        save(fig,number)

def unit_axes(ax):
    t=np.linspace(0,2*np.pi,200);ax.plot(np.cos(t),np.sin(t),color='#dddddd',lw=.6,zorder=0)
    ax.axhline(0,color='#eeeeee',lw=.5,zorder=0);ax.axvline(0,color='#eeeeee',lw=.5,zorder=0)
    ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),xlabel='Unit update along Linear',ylabel='Common orthogonal PC1',aspect='equal')

def case_figure():
    plt=style();fig,axes=plt.subplots(1,3,figsize=(10.5,3.8));src=rows(PAPER/'data/training_validation_test.csv')
    for j,part in enumerate(['validation','benchmark']):
        for k,w in enumerate([0.,.01]):
            r=next(r for r in src if r['model']=='qwen3' and r['concept']=='hallucination' and r['partition']==part and r['method']==f'lambda_{w}');v=float(r['positive_delta_pp']);lo=float(r['ci_low']);hi=float(r['ci_high'])
            axes[0].errorbar(j+(k-.5)*.2,v,yerr=[[v-lo],[hi-v]],fmt='o',color=WC[w],capsize=3,label=lam(w) if j==0 else None)
    axes[0].set(xticks=[0,1],xticklabels=['Validation','Official Test'],ylabel=r'Positive $\Delta P$ (pp)',ylim=(0,48),xlim=(-.5,1.5));axes[0].legend(fontsize=8);decorate(axes[0])
    val=next(r for r in rows(PAPER/'data/paired_validation_intervals.csv') if r['model']=='qwen3' and r['concept']=='hallucination' and float(r['weight'])==.01)
    test=next(r for r in src if r['model']=='qwen3' and r['concept']=='hallucination' and r['partition']=='benchmark' and r['method']=='lambda_0.01')
    axes[0].text(.02,.10,f"Val. paired gain: {float(val['delta_vs_lambda0_pp']):+.2f} [{float(val['paired_ci_low']):+.2f}, {float(val['paired_ci_high']):+.2f}]",transform=axes[0].transAxes,fontsize=6.5)
    axes[0].text(.02,.04,f"Test paired gain: {float(test['vs_lambda0_pp']):+.2f} [{float(test['vs_lambda0_ci_low']):+.2f}, {float(test['vs_lambda0_ci_high']):+.2f}]",transform=axes[0].transAxes,fontsize=6.5)
    ds=rows(PAPER/'data/qwen_hallucination_displacement.csv')
    for k,w in enumerate([0.,.01]):
        r=next(r for r in ds if float(r['weight'])==w);x=float(r['rms']);y=float(r['mean_cosine_to_linear'])
        axes[1].scatter(x,y,s=40,color=WC[w]);axes[1].text(.04,.18-k*.1,lam(w)+f': RMS {x:.3f}, cosine {y:.4f}',transform=axes[1].transAxes,fontsize=7,color=WC[w])
    axes[1].set(xlim=(0,300),ylim=(0,1.06),xlabel='Validation displacement RMS',ylabel='Mean update cosine with Linear');decorate(axes[1])
    contrasts=rows(PAPER/'data/qwen_factorial_contrasts.csv')
    for i,key in enumerate(['symmetric_direction_contribution','symmetric_magnitude_contribution']):
        r=next(r for r in contrasts if r['contrast']==key);v=float(r['mean_pp']);lo=float(r['ci_low']);hi=float(r['ci_high'])
        axes[2].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE if i==0 else '#777777',capsize=3)
    axes[2].set(yticks=[0,1],yticklabels=['Direction','Magnitude'],ylim=(-.5,1.5),xlabel='Diagnostic contribution (pp)');axes[2].axvline(0,color='k',lw=.6)
    for i,ax in enumerate(axes):badge(ax,i)
    fig.suptitle('Qwen3 Hallucination · verified case study\nPointwise 95% CIs; Official Test is exploratory. Crossover is a diagnostic, not a new method.',fontsize=10)
    save(fig,20)

def response_panels(stats,statuses):
    plt=style();complete=[];draft=[]
    def full(m,c):
        methods=['Linear']+[f'lambda_{w}' for w in sorted({0.,selected(m)[c]})]
        return all(len([r for r in stats if r['model']==m and r['concept']==c and r['method']==k and r['multiplier']==a])==1 for k in methods for a in [-1.,-.5,0.,.5,1.])
    def draw(ax,m,c):
        if not full(m,c):
            ax.text(.5,.5,'Not yet complete',ha='center',va='center',transform=ax.transAxes,color='#888888')
        else:
            methods=['Linear']+[f'lambda_{w}' for w in sorted({0.,selected(m)[c]})]
            bases=[]
            for method in methods:
                rr=sorted([r for r in stats if r['model']==m and r['concept']==c and r['method']==method],key=lambda r:r['multiplier'])
                color=BLUE if method=='Linear' else WC[float(method.split('_')[1])];label='CAA Linear' if method=='Linear' else lam(float(method.split('_')[1]))
                ax.plot([r['multiplier'] for r in rr],[r['probability'] for r in rr],color=color,lw=1.2,label=label)
                ss=[r for r in rr if r['multiplier']!=0]
                ax.errorbar([r['multiplier'] for r in ss],[r['probability'] for r in ss],yerr=[[r['probability']-r['ci_low'] for r in ss],[r['ci_high']-r['probability'] for r in ss]],fmt='o',color=color,ms=3,capsize=2,lw=.9)
                bases.append(next(r for r in rr if r['multiplier']==0))
            assert np.ptp([r['probability'] for r in bases])<1e-12
            b=bases[0];ax.errorbar(0,b['probability'],yerr=[[b['probability']-b['ci_low']],[b['ci_high']-b['probability']]],fmt='D',color='k',ms=3,capsize=2,label='No-hook')
        ax.set(title=NAMES[c]+'\n'+lam(selected(m)[c])+' selected',xlabel='Strength multiplier',ylabel='Target probability',ylim=(-.02,1.02),xticks=[-1,-.5,0,.5,1]);decorate(ax)
    for m,n in [('qwen3',10),('llama3b',11),('llama2',21)]:
        ready=all(full(m,c) for c in NAMES);fig,axes=plt.subplots(2,4,figsize=(11,6))
        for i,(ax,c) in enumerate(zip(axes.flat,NAMES)):draw(ax,m,c);badge(ax,i)
        ax=axes.flat[-1];ax.axis('off');ws=sorted({0.}|set(selected(m).values()))
        handles=[Line2D([],[],color=BLUE,marker='o',label='CAA Linear')]+[Line2D([],[],color=WC[w],marker='o',label='Curveball, '+lam(w)) for w in ws]+[Line2D([],[],color='k',marker='D',ls='',label='No-hook')]
        ax.legend(handles=handles,loc='center',fontsize=7)
        ax.text(.5,.03,'Kernel weight '+r'$\lambda$'+' is not strength.\nBars: pointwise 95% CIs.',ha='center',va='bottom',fontsize=7,transform=ax.transAxes)
        fig.suptitle(('DRAFT · ' if not ready else '')+SHORT[m]+' · validation response curves\nFive fixed strengths; each method retains its own frozen native step.',fontsize=11)
        save(fig,n,draft=not ready);(complete if ready else draft).append(n)
    cases=[('qwen3','hallucination'),('llama3b','hallucination'),('llama2','myopic-reward')];ready=all(full(m,c) for m,c in cases)
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.8))
    for i,(ax,(m,c)) in enumerate(zip(axes,cases)):
        draw(ax,m,c);ax.set_title(SHORT[m]+'\n'+NAMES[c]+' · '+lam(selected(m)[c]));badge(ax,i)
        if full(m,c):ax.legend(fontsize=6.5,loc='upper left')
    fig.suptitle(('DRAFT · ' if not ready else '')+'Representative validation responses · true no-hook at zero\nEqual multipliers do not imply equal activation displacement.',fontsize=10)
    save(fig,3,draft=not ready);(complete if ready else draft).append(3)
    save_route=dict(complete=sorted(complete),draft=sorted(draft))
    save_json(PAPER/'data/figure_routing.json',save_route)

# Keep JSON writing distinct from figure saving.
from artifact_common import save as save_json
