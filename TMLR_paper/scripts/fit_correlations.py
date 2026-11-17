"""In-sample companion to the validation paired-target correlation analysis."""
import joblib, torch
from geometry import corr_stats
from artifact_common import *

def main():
    torch.set_num_threads(2)
    CB = fit_class()
    stats, observations = [], []
    for model in MODELS:
        cfg = config(model)
        steps = load(base(model) / 'fits' / model / 'positive_steps.json')['steps']
        for ci, c in enumerate(NAMES):
            cache = torch.load(Path(cfg['activation_sources'][model]) / f'train_activations_{c}.pt', weights_only=True)
            rr = split_rows(c, 'fit')
            ii = [r['index'] for r in rr]
            h = cache['negative'][ii].float()
            distance = (cache['positive'][ii].double() - h.double()).norm(dim=-1).numpy()
            groups = [r['group_id'] for r in rr]
            for w in sorted({0., .01, 1., selected(model)[c]}):
                cb = CB.restore(joblib.load(fit_path(model, c, w)))
                assert torch.equal(cb.intervention(h[:4], 0), h[:4])
                magnitudes = []
                for batch in h.split(64):
                    magnitudes.extend((cb.intervention(batch, float(steps[c]['1'])).double() - batch.double()).norm(dim=-1).tolist())
                mag = np.array(magnitudes)
                stats.append(dict(model=model, concept=c, behavior=NAMES[c], partition='fit', weight=w,
                                  selected=w == selected(model)[c], **corr_stats(mag, distance, groups, 20261111 + ci)))
                observations.extend(dict(model=model, concept=c, partition='fit', weight=w, index=r['index'],
                                         group_id=r['group_id'], displacement=float(mag[j]), paired_target_distance=float(distance[j]))
                                    for j, r in enumerate(rr))
            print(json.dumps(dict(stage='fit_geometry', model=model, concept=c)), flush=True)
    ordered = sorted([(i, r['permutation_p']) for i, r in enumerate(stats) if r['permutation_p'] is not None], key=lambda x: x[1])
    last = 0.
    for rank, (i, p) in enumerate(ordered):
        last = max(last, min(1., (len(ordered) - rank) * p))
        stats[i]['holm_adjusted_p'] = last
    write_csv(PAPER / 'data/geometric_diagnostics/fit_magnitude_correlation.csv', stats)
    write_csv(PAPER / 'data/geometric_diagnostics/fit_paired_activation_updates.csv', observations)
    save(PAPER / 'data/geometric_diagnostics/fit_correlation_completion.json', dict(
        complete=True, conditions=len(stats), observation_rows=len(observations), official_test_used=False,
        interpretation='In-sample genuine paired training activations. Same diagnostic weights and native steps as validation geometry; no refitting or selection. Holm family is separate from validation.'))

if __name__ == '__main__':
    main()
