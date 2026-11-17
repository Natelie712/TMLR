"""Render the released main table only; manuscript figures are withdrawn."""
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'TMLR_paper/scripts'))
import artifact_common as common
import format_tables

DEST=ROOT/'reproduced/TMLR_paper'
(DEST/'data').mkdir(parents=True,exist_ok=True)
(DEST/'tables').mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'TMLR_paper/data/main_results.csv',DEST/'data/main_results.csv')
common.PAPER=DEST
format_tables.PAPER=DEST
format_tables.format_main()
assert (DEST/'tables/main_results.tex').read_bytes()==(ROOT/'TMLR_paper/tables/main_results.tex').read_bytes()
print('Re-rendered identical main table at',DEST/'tables/main_results.tex')
print('Figures are withdrawn; none were generated. Existing local outputs from earlier releases are not reviewed figures.')
