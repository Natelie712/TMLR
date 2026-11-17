"""Re-render completed artifacts from released caches into an isolated directory."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'TMLR_paper/scripts'))
import artifact_common as common
import refined_figures as figures
import format_tables

DEST=ROOT/'reproduced/TMLR_paper'
DEST.mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT/'TMLR_paper/data',DEST/'data',dirs_exist_ok=True)
shutil.copytree(ROOT/'TMLR_paper/tables',DEST/'tables',dirs_exist_ok=True)
(DEST/'figures').mkdir(exist_ok=True)
common.PAPER=DEST
figures.PAPER=DEST
figures.P=DEST/'data/geometric_diagnostics'
format_tables.PAPER=DEST
format_tables.format_main()
figures.main_results_figure()
figures.geometry_figures()
figures.case_figure()
with (DEST/'data/response_summary.csv').open() as f:
    stats=list(csv.DictReader(f))
for row in stats:
    for key in ['multiplier','probability','ci_low','ci_high']:
        row[key]=float(row[key])
figures.response_panels(stats,{m:{'status':'complete'} for m in common.MODELS})
pdfs=sorted((DEST/'figures').glob('*.pdf'))
assert len(pdfs)==18, len(pdfs)
print('Re-rendered main table and',len(pdfs),'figures at',DEST)
