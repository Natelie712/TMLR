"""Durable Slurm recovery for this frozen experiment; never retries science failures."""
import fcntl, getpass, json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'TMLR_paper/data/execution'
STATE = OUT / 'automatic_recovery.json'
GPU_SCRIPT = ROOT / 'TMLR_paper/scripts/responses.slurm'
WATCH_SCRIPT = ROOT / 'TMLR_paper/scripts/recovery_controller.slurm'
GPU_TYPES = ['a40', 'a6000', 'l40']  # 48GB, compatible with the frozen CUDA build.
RETRY = {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'TIMEOUT'}
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'}

def command(args):
    return subprocess.check_output(args, cwd=ROOT, universal_newlines=True, stderr=subprocess.STDOUT, timeout=60).strip()

def save(state):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(STATE)

def event(state, kind, **kw):
    record = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), event=kind, **kw)
    state.setdefault('events', []).append(record)
    print(json.dumps(record), flush=True)
    save(state)

def active_jobs():
    result = command(['squeue','--noheader','--user='+getpass.getuser(),'--name=tmlr-response-curves','--format=%i|%T'])
    found = []
    for line in result.splitlines():
        jid, status = line.split('|')
        info = command(['scontrol','show','job',jid.strip(),'-o'])
        if 'Command='+str(GPU_SCRIPT) in info:
            found.append((jid.strip(),status.strip()))
    return found

def accounting(jid):
    result = command(['sacct','-n','-P','-j',jid,'--format=JobIDRaw,State,NodeList'])
    for line in result.splitlines():
        fields=line.split('|')
        if fields[0]==jid:
            return fields[1].split()[0].rstrip('+'), fields[2]
    return 'UNKNOWN',''

def science_error(jid):
    p=OUT/f'responses-{jid}.err'
    return p.exists() and 'Traceback (most recent call last)' in p.read_text(errors='replace')

def decision(status, traceback_present=False):
    if traceback_present: return 'stop'
    if status in ACTIVE or status=='UNKNOWN': return 'wait'
    if status in RETRY: return 'retry'
    if status=='COMPLETED': return 'verify'
    return 'stop'  # Includes deliberate cancellation, OOM, assertion/code failure.

def finished():
    p=ROOT/'TMLR_paper/data/artifact_status.json'
    return p.exists() and json.loads(p.read_text()).get('response_complete',False)

def submit_gpu(state, previous_node):
    # Query twice: never launch a duplicate after a scheduler/accounting race.
    if active_jobs():return
    if state.get('submission_uncertain'):
        raise RuntimeError('Previous sbatch outcome uncertain; inspect queue before retrying submission.')
    previous=state.get('gpu_type','a40')
    gpu=GPU_TYPES[(GPU_TYPES.index(previous)+1)%len(GPU_TYPES)]
    args=['sbatch','--parsable','--requeue','--time=02:00:00','--partition=notchpeak-gpu-guest',
          '--account=owner-gpu-guest','--qos=notchpeak-gpu-guest','--gres=gpu:'+gpu+':1',
          '--output='+str(OUT/'responses-%j.out'),'--error='+str(OUT/'responses-%j.err')]
    if previous_node and re.fullmatch(r'notch\d+',previous_node):args += ['--exclude='+previous_node]
    args += [str(GPU_SCRIPT)]
    state['submission_uncertain']=True;save(state)
    jid=command(args).split(';')[0]
    assert jid.isdigit(),jid
    state.update(job_id=jid,gpu_type=gpu,submission_uncertain=False,retries=state.get('retries',0)+1,status='monitoring')
    event(state,'checkpoint_resume_submitted',job_id=jid,gpu_type=gpu,excluded_previous_node=previous_node)

def main():
    lock=(OUT/'automatic_recovery.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text())
    if state.get('status') in ['complete','stopped']:return
    me=os.environ['SLURM_JOB_ID'];state['controller_job_id']=me
    # A lightweight CPU successor recovers even if this controller is preempted.
    successor=command(['sbatch','--parsable','--dependency=afterany:'+me,
                       '--output='+str(OUT/'controller-%j.out'),'--error='+str(OUT/'controller-%j.err'),str(WATCH_SCRIPT)]).split(';')[0]
    assert successor.isdigit()
    state['controller_successor']=successor;event(state,'controller_started',job_id=me,successor=successor)
    start=time.monotonic()
    try:
        while time.monotonic()-start < 7*3600:
            try:
                if json.loads(STATE.read_text()).get('status')=='stopped':
                    state['status']='stopped';event(state,'recovery_disabled_by_user');break
                if time.time()>state['deadline_epoch'] or state.get('retries',0)>=12:
                    state['status']='stopped';event(state,'retry_window_exhausted');break
                jobs=active_jobs()
                if len(jobs)>1:
                    state['status']='stopped';event(state,'multiple_active_experiment_jobs',jobs=jobs);break
                if jobs:
                    jid,status=jobs[0]
                    if jid!=state['job_id']:
                        state.update(job_id=jid,submission_uncertain=False)
                        event(state,'existing_experiment_job_adopted',job_id=jid)
                else:
                    status,node=accounting(state['job_id'])
                    action=decision(status,science_error(state['job_id']))
                    if action=='retry':
                        event(state,'infrastructure_interruption',job_id=state['job_id'],scheduler_state=status)
                        submit_gpu(state,node)
                    elif action=='verify':
                        state['status']='complete' if finished() else 'stopped'
                        event(state,'experiment_'+state['status'],job_id=state['job_id']);break
                    elif action=='stop':
                        state['status']='stopped';event(state,'manual_or_scientific_failure_requires_review',job_id=state['job_id'],scheduler_state=status);break
                time.sleep(60)
            except RuntimeError as error:
                state['status']='stopped';event(state,'ambiguous_submission_requires_review',detail=str(error));break
            except (subprocess.SubprocessError,OSError,ValueError) as error:
                # Scheduler outages do not authorize duplicate submissions.
                event(state,'scheduler_query_error',detail=str(error));time.sleep(60)
    finally:
        if state.get('status') in ['complete','stopped']:
            command(['scancel',successor])  # Only this controller's own successor.
        else:event(state,'controller_handoff',successor=successor)

if __name__=='__main__':main()
