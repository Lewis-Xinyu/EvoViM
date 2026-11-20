"""Test a retrieved similarity against spatial samples outside its fitting disk.

Transforms are supplied by explicit retrieval, never refitted on these samples.
No threshold declares a match, no region detector or action is implemented.
"""
import math
import numpy as np
from scipy import ndimage as ndi
from patch_match import pixels


def check_alignment(query_image, source_image, transform, query_box=None):
    query = pixels(query_image, 256)
    source = ndi.gaussian_filter(pixels(source_image, 256), (.5, .5, 0))
    qh, qw = query.shape[:2]
    sh, sw = source.shape[:2]
    qe, se = max(qh, qw), max(sh, sw)
    # Uniform normalized sample positions; resolution controls compute only.
    axis = (np.arange(48, dtype=float) + .5) / 48
    uy, ux = np.meshgrid(axis, axis, indexing='ij')
    qx, qy = ux*(qw-1), uy*(qh-1)
    x, y = (qx-(qw-1)/2)/qe, (qy-(qh-1)/2)/qe
    ct, st = math.cos(transform['angle']), math.sin(transform['angle'])
    sx = (transform['cx']+transform['scale']*(ct*x-st*y))*se
    sy = (transform['cy']+transform['scale']*(st*x+ct*y))*se
    valid = (sx>=0)&(sy>=0)&(sx<=sw-1)&(sy<=sh-1)
    # Outside positions are uncovered, never scored as clamped edge pixels.
    target = np.stack([ndi.map_coordinates(query[:,:,c], [qy,qx], order=1)
                       for c in range(3)], axis=-1)
    values = np.stack([ndi.map_coordinates(source[:,:,c], [sy,sx], order=1,
                                          mode='constant', cval=0)
                       for c in range(3)], axis=-1)
    central = x*x+y*y <= (.36*min(qh,qw)/qe)**2
    calibration = valid & central
    offset = ((values.astype(np.float64)-target)[calibration].mean(axis=0)
              if calibration.any() else None)
    box = query_box or [0., 0., 1., 1.]
    region = (ux>=box[0])&(uy>=box[1])&(ux<=box[2])&(uy<=box[3])

    def stats(mask):
        eligible, count = int(mask.sum()), int((mask & valid).sum())
        report = {'samples':eligible, 'covered_samples':count,
                  'coverage':round(count/eligible,5) if eligible else 0.}
        if count == 0:
            return dict(report, correlation=None, query_contrast=None,
                        source_contrast=None, centred_rmse=None,
                        calibrated_rmse=None, mean_colour_error=None)
        # Accurate reductions keep a constant float32 patch from acquiring
        # spurious variance solely through summation error.
        q, r = target[mask & valid].astype(np.float64), values[mask & valid].astype(np.float64)
        a, b = r-r.mean(axis=0), q-q.mean(axis=0)
        energy = float(np.sqrt((a*a).sum()*(b*b).sum()))
        corr = float((a*b).sum()/energy) if energy>1e-12 else None
        return dict(report,
                    correlation=round(corr,5) if corr is not None else None,
                    query_contrast=round(float(np.sqrt((b*b).mean())),5),
                    source_contrast=round(float(np.sqrt((a*a).mean())),5),
                    centred_rmse=round(float(np.sqrt(((a-b)**2).mean())),5),
                    calibrated_rmse=(round(float(np.sqrt(((r-q-offset)**2).mean())),5)
                                     if offset is not None else None),
                    mean_colour_error=round(float(np.sqrt(((r.mean(axis=0)-q.mean(axis=0))**2).mean())),5))

    cells = []
    for row in range(3):
        for col in range(3):
            cell = region & (uy>=row/3)&(uy<(row+1)/3)&(ux>=col/3)&(ux<(col+1)/3)
            if cell.any():
                cells.append({'row':row,'column':col, **stats(cell)})
    return {'query_box':list(box), 'transform':dict(transform),
            'region':stats(region), 'centre':stats(region & central),
            'outside_centre':stats(region & ~central), 'cells':cells,
            'calibration_samples':int(calibration.sum()),
            'centre_colour_offset':offset.tolist() if offset is not None else None,
            'interpretation':'Fixed retrieved transform; no fitting to outer pixels. Correlation is offset-invariant. calibrated_rmse uses only the central-disk colour offset. Coverage is geometric, not informative area. Blank areas and query padding can disagree for a correct source; compare informative cells and alternative hypotheses. No metric is a correctness probability.'}
