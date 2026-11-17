"""Separate durable recovery controller for the negative-lambda search."""
import json,getpass
import recovery_controller as r
r.OUT=r.ROOT/'TMLR_paper/data/negative_lambda_search/execution'
r.STATE=r.OUT/'automatic_recovery.json'
r.GPU_SCRIPT=r.ROOT/'TMLR_paper/scripts/negative_search.slurm'
r.WATCH_SCRIPT=r.ROOT/'TMLR_paper/scripts/negative_recovery.slurm'

def active_jobs():
    result=r.command(['squeue','--noheader','--user='+getpass.getuser(),'--name=tmlr-negative-lambda','--format=%i|%T'])
    found=[]
    for line in result.splitlines():
        jid,status=line.split('|');info=r.command(['scontrol','show','job',jid.strip(),'-o'])
        if 'Command='+str(r.GPU_SCRIPT) in info:found.append((jid.strip(),status.strip()))
    return found

def finished():
    path=r.OUT.parent/'completion.json'
    return path.exists() and json.loads(path.read_text()).get('complete',False)

r.active_jobs=active_jobs
r.finished=finished
if __name__=='__main__':r.main()
