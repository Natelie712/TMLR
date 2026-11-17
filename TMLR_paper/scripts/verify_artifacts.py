"""Check source integrity, table arithmetic, split identity and artifact coverage."""
from artifact_common import *
from importlib.metadata import version
import platform

def main():
    frozen=load(PAPER/'data/revision_config.json')['frozen_inputs']
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    manifest=load(PAPER/'data/table_source_manifest.json')
    for p,h in manifest['files'].items():assert sha(ROOT/p)==h,p
    mainrows=rows(PAPER/'data/main_results.csv');assert len(mainrows)==21
    assert len({(r['model'],r['concept']) for r in mainrows})==21
    for r in mainrows:
        assert abs(float(r['curveball_delta_pp'])-float(r['linear_delta_pp'])-float(r['difference_pp']))<1e-8
        assert abs(100*(float(r['positive_probability'])-float(r['nohook_probability']))-float(r['curveball_delta_pp']))<1e-8
        assert float(r['selected_lambda'])==selected(r['model'])[r['concept']]
        assert float(r['difference_ci_low'])<=float(r['difference_ci_high'])
    ds=rows(PAPER/'data/dataset_statistics.csv')
    assert sum(int(r['fit_questions']) for r in ds)==3906
    assert sum(int(r['validation_questions']) for r in ds)==912
    assert sum(int(r['benchmark_questions']) for r in ds)==350
    assert load(PAPER/'data/global_split_leakage_check.json')['passed']
    for r in ds:
        assert int(r['fit_total_activation_samples'])==2*int(r['fit_questions'])
    eig=rows(PAPER/'data/geometric_diagnostics/kernel_variance_reconstruction.csv')
    assert len(eig)==189
    for r in eig:assert 0<=float(r['retained_kernel_variance_fraction'])<=1
    assert load(PAPER/'data/geometric_diagnostics/fit_correlation_completion.json')['complete']
    publication=load(PAPER/'data/publication_checks.json');assert publication['latex_compile_passed']
    assert not publication['layout_warnings'],publication['layout_warnings']
    assert load(PAPER/'data/figure_scientific_checks.json')['all_passed']
    route=load(PAPER/'data/figure_routing.json')
    import zipfile
    with zipfile.ZipFile(PAPER/'data/overleaf_artifacts.zip') as bundle:
        names=set(bundle.namelist())
        assert not any('/drafts/' in name for name in names)
        for n in route['draft']:
            assert not (PAPER/f'figures/figure_{n}.pdf').exists()
            assert (PAPER/f'figures/drafts/figure_{n}.pdf').exists()
            assert f'figures/figure_{n}.pdf' not in names
        for p in (PAPER/'figures').glob('figure_*.pdf'):
            assert str(p.relative_to(PAPER)) in names
    save(PAPER/'data/integrity_checks.json',dict(passed=True,protected_input_files=len(frozen),
         historical_table_sources_verified=len(manifest['files']),main_conditions=21,kernel_fits=189,
         table_arithmetic_passed=True,split_counts_passed=True,source_group_disjoint=True,
         latex_compiled_without_overfull_boxes=True,scope='Current CPU artifacts and protected source checks. GPU replay is reported separately.'))
    save(PAPER/'data/analysis_source_manifest.json',dict(files={str(p.relative_to(PAPER)):sha(p) for p in sorted((PAPER/'scripts').iterdir()) if p.is_file()},
         python=platform.python_version(),packages={p:version(p) for p in ['torch','transformers','numpy','scipy','scikit-learn','joblib','matplotlib','markdown-it-py']}))
    print('TMLR artifact integrity checks passed.',flush=True)

if __name__=='__main__':main()
