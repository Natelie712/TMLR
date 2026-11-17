"""Run the frozen search inside an authorized allocation with durable recovery."""
import fcntl,json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'TMLR_paper/data/negative_lambda_search/execution'
STATE=OUT/'allocation_recovery.json'
WATCH=ROOT/'TMLR_paper/scripts/allocated_negative_controller.slurm'
RUN=ROOT/'TMLR_paper/scripts/negative_search.slurm'
INFRA={'PREEMPTED','NODE_FAIL','BOOT_FAIL','TIMEOUT'}

def command(args):return subprocess.check_output(args,cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=60).strip()
def save(s):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)
def event(s,kind,**kw):
    e=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw)
    s.setdefault('events',[]).append(e);save(s);print(json.dumps(e),flush=True)
def allocation_state(s):
    text=command(['sacct','-M',s['cluster'],'-n','-P','-j',s['allocation'],'--format=JobIDRaw,State'])
    for line in text.splitlines():
        a=line.split('|')
        if a[0]==s['allocation']:return a[1].split()[0].rstrip('+')
    return 'UNKNOWN'
def steps(s):
    text=command(['squeue','-M',s['cluster'],'--steps','--jobs='+s['allocation'],'--noheader','--format=%i|%j'])
    return [line.split('|')[0].strip() for line in text.splitlines() if '|tmlr-negative-search-step' in line]
def complete():
    p=OUT.parent/'completion.json'
    return p.exists() and json.loads(p.read_text()).get('complete',False)
def science_failure(s):
    for name in s.get('run_logs',[]):
        p=OUT/name
        if p.exists() and 'Traceback (most recent call last)' in p.read_text(errors='replace'):return True
    reviewed=s.get('reviewed_failure_status_hashes',{})
    return any(json.loads(p.read_text()).get('status')=='failed' and
               reviewed.get(p.name)!=hashlib.sha256(p.read_bytes()).hexdigest()
               for p in OUT.parent.glob('*_status.json'))

def fallback(s,reason):
    # Preserve the user's IDE allocation. Only this experiment gets a new job.
    if command(['squeue','--noheader','--user='+os.environ['USER'],'--name=tmlr-negative-lambda','--format=%i']):
        raise RuntimeError('Existing negative batch job found; refusing duplicate fallback')
    if s.get('fallback_submission_uncertain'):raise RuntimeError('Ambiguous previous submission; manual reconciliation required')
    s['fallback_submission_uncertain']=True;event(s,'fallback_intent',reason=reason)
    jid=command(['sbatch','--parsable','--requeue','--partition=notchpeak-gpu-guest','--account=owner-gpu-guest','--qos=notchpeak-gpu-guest','--gres=gpu:a40:1','--output='+str(OUT/'responses-%j.out'),'--error='+str(OUT/'responses-%j.err'),str(RUN)]).split(';')[0]
    assert jid.isdigit()
    p=OUT/'automatic_recovery.json';r=json.loads(p.read_text())
    r.update(status='monitoring',job_id=jid,gpu_type='a40',submission_uncertain=False,deadline_epoch=s['deadline_epoch'])
    r['events'].append(dict(event='granite_infrastructure_fallback',job_id=jid,reason=reason,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
    controller=command(['sbatch','--parsable','--output='+str(OUT/'controller-%j.out'),'--error='+str(OUT/'controller-%j.err'),str(ROOT/'TMLR_paper/scripts/negative_recovery.slurm')]).split(';')[0]
    s.update(status='transferred',fallback_submission_uncertain=False,fallback_job=jid,fallback_controller=controller)
    event(s,'fallback_submitted',job_id=jid,controller=controller)

def main():
    lock=(OUT/'allocation_recovery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    s=json.loads(STATE.read_text())
    if s['status'] in ['complete','stopped','transferred']:return
    me=os.environ['SLURM_JOB_ID'];previous=s.get('controller_job')
    if previous and previous!=me:
        history=command(['sacct','-n','-P','-j',previous,'--format=JobIDRaw,State'])
        if any(line.split('|')[:2]==[previous,'CANCELLED'] or (line.startswith(previous+'|') and '|CANCELLED ' in line) for line in history.splitlines()):
            s['status']='stopped';event(s,'controller_deliberately_cancelled');return
    successor=command(['sbatch','--parsable','--dependency=afterany:'+me,'--output='+str(OUT/'allocation-controller-%j.out'),'--error='+str(OUT/'allocation-controller-%j.err'),str(WATCH)]).split(';')[0]
    assert successor.isdigit();s.update(controller_job=me,controller_successor=successor)
    event(s,'controller_started',job_id=me,previous=previous,successor=successor)
    proc=None;began=time.monotonic()
    try:
        while True:
            try:
                if json.loads(STATE.read_text()).get('status')=='stopped':s['status']='stopped';break
                if complete():s['status']='complete';event(s,'experiment_complete');break
                if science_failure(s):s['status']='stopped';event(s,'scientific_failure_requires_review');break
                state=allocation_state(s)
                if state in INFRA:
                    fallback(s,state);break
                if state in ['CANCELLED','COMPLETED','FAILED','OUT_OF_MEMORY']:
                    s['status']='stopped';event(s,'allocation_ended_requires_review',state=state);break
                if time.time()>s['deadline_epoch']:
                    s['status']='stopped';event(s,'recovery_window_exhausted');break
                if proc is not None:
                    rc=proc.poll()
                    if rc is None:time.sleep(30);continue
                    event(s,'step_driver_exit',returncode=rc);proc=None
                    if complete():continue
                    if allocation_state(s) in INFRA:continue
                    s['status']='stopped';event(s,'step_failure_requires_review',returncode=rc);break
                active=steps(s)
                if active:
                    # A previous CPU controller can die while its GPU step survives.
                    if len(active)>1:raise RuntimeError('Duplicate experiment steps detected')
                    time.sleep(30);continue
                if state not in ['RUNNING','UNKNOWN']:time.sleep(30);continue
                if state=='UNKNOWN':time.sleep(30);continue
                if time.monotonic()-began>7*3600:event(s,'controller_handoff');break
                stamp=str(time.time_ns());out=OUT/f'allocated-{stamp}.out';err=OUT/f'allocated-{stamp}.err'
                s.setdefault('run_logs',[]).extend([out.name,err.name])
                event(s,'starting_allocated_step',allocation=s['allocation'],cluster=s['cluster'],stdout=out.name,stderr=err.name)
                args=['srun','-M',s['cluster'],'--jobid='+s['allocation'],'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=64G','--gres=gpu:1','--cpu-bind=none','--export=ALL','--job-name=tmlr-negative-search-step','--output='+str(out),'--error='+str(err),'/bin/bash',str(RUN)]
                env=dict(os.environ,PATH=os.environ.get('PATH','/usr/bin:/bin'),SLURM_EXPORT_ENV='ALL')
                proc=subprocess.Popen(args,cwd=ROOT,env=env)
                s['srun_pid']=proc.pid;save(s)
            except (subprocess.SubprocessError,OSError,ValueError) as e:
                event(s,'scheduler_query_error',detail=str(e));time.sleep(30)
    except RuntimeError as e:
        s['status']='stopped';event(s,'safety_stop',detail=str(e));raise
    finally:
        if s['status'] in ['complete','stopped','transferred']:command(['scancel',successor])

if __name__=='__main__':main()
