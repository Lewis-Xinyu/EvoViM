"""Independent dense, rotation/scale tolerant retrieval from retained pixels.

No detector selects a product region: the full reference image is searched.
Discretisation controls compute, not episode-specific acceptance thresholds.
The central disk avoids flat borders introduced by rotating/cropping a query.
Returned correlations are evidence, never calibrated probabilities or actions.
"""
import math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import fft
from scipy.optimize import least_squares


def pixels(image, side):
    im = image.convert('RGB')
    im.thumbnail((side, side), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def prepare_query(image):
    q = pixels(image, 128)
    extent = min(q.shape[:2])
    radius = .36 * extent
    # Translation, scale and rotation lattice; positions are searched by FFT.
    sizes = sorted(set(round(x) for x in np.geomspace(8, 96, 12)))
    angles = np.arange(12) * (2 * np.pi / 12)
    programs = []
    shape = (192, 192)
    for size in sizes:
        axis = (np.arange(size) - (size-1)/2) / (size/2)
        y, x = np.meshgrid(axis, axis, indexing='ij')
        mask = (x*x+y*y <= 1).astype(np.float32)
        count = float(mask.sum())
        smooth = ndi.gaussian_filter(q, (max(0, radius/size*.5), max(0, radius/size*.5), 0))
        kernels = []
        energies = []
        for angle in angles:
            ct, st = np.cos(angle), np.sin(angle)
            qx = (ct*x+st*y)*radius + (q.shape[1]-1)/2
            qy = (-st*x+ct*y)*radius + (q.shape[0]-1)/2
            patch = np.stack([ndi.map_coordinates(smooth[:,:,c], [qy,qx], order=1, mode='nearest') for c in range(3)])
            means = (patch*mask).sum(axis=(1,2))/count
            centred = (patch-means[:,None,None])*mask
            energies.append(float((centred**2).sum()))
            kernels.append(centred[:,::-1,::-1])
        programs.append({'size':size, 'angles':angles, 'count':count,
                         'mask_fft':fft.rfft2(mask, s=shape),
                         'kernels_fft':fft.rfft2(np.asarray(kernels), s=shape),
                         'energies':np.asarray(energies,dtype=np.float32)})
    return {'programs':programs, 'shape':shape, 'radius_fraction':radius/max(q.shape[:2]),
            'full':pixels(image,256)}


def search(prepared, image, local_proposal=None):
    r = pixels(image,96)
    h,w = r.shape[:2]
    shape = prepared['shape']
    rf = fft.rfft2(r.transpose(2,0,1), s=shape)
    r2f = fft.rfft2((r**2).sum(axis=2), s=shape)
    best = None
    for p in prepared['programs']:
        d = p['size']
        if d>min(h,w): continue
        sums = fft.irfft2(rf*p['mask_fft'],s=shape)[:,d-1:h,d-1:w]
        squares = fft.irfft2(r2f*p['mask_fft'],s=shape)[d-1:h,d-1:w]
        variance = np.maximum(0, squares-(sums**2).sum(axis=0)/p['count'])
        cross = fft.irfft2((rf[None]*p['kernels_fft']).sum(axis=1),s=shape)[:,d-1:h,d-1:w]
        denom = np.sqrt(p['energies'][:,None,None]*variance[None])
        scores = np.divide(cross,denom,out=np.zeros_like(cross),where=denom>1e-6)
        a,y,x = np.unravel_index(np.argmax(scores),scores.shape)
        value = float(scores[a,y,x])
        if best is None or value>best['correlation']:
            best = {'correlation':value, 'cx':(x+(d-1)/2)/max(h,w),
                    'cy':(y+(d-1)/2)/max(h,w),
                    'scale':d/(2*prepared['radius_fraction']*max(h,w)),
                    'angle':float(p['angles'][a])}
    if best is None: return {'correlation':0., 'box':None, 'status':'reference too small'}
    reference = pixels(image,256)
    result = refine(prepared['full'],reference,best)
    result['proposal'] = 'full-image correlation search'
    if local_proposal is not None:
        local = refine(prepared['full'],reference,dict(local_proposal,correlation=0.))
        local['proposal'] = 'local-feature geometry, independently checked with dense pixels'
        if local['correlation']*local['coverage'] > result['correlation']*result['coverage']:
            result = local
    return result


def refine(query, reference, proposal):
    """Fit one proposed similarity using hundreds of distinct coloured pixels."""
    qh,qw = query.shape[:2]
    rh,rw = reference.shape[:2]
    extent = max(qh,qw)
    rextent = max(rh,rw)
    axis = np.linspace(-.36,.36,32) * min(qh,qw)/extent
    yy,xx = np.meshgrid(axis,axis,indexing='ij')
    keep = xx*xx+yy*yy <= (.36*min(qh,qw)/extent)**2
    xy = np.stack([xx[keep],yy[keep]])
    qx = xy[0]*extent+(qw-1)/2
    qy = xy[1]*extent+(qh-1)/2
    target = np.stack([ndi.map_coordinates(query[:,:,c],[qy,qx],order=1) for c in range(3)],axis=1)
    reference = ndi.gaussian_filter(reference,(.5,.5,0))
    def sample(v):
        cx,cy,logs,angle = v
        scale = np.exp(logs)
        ct,st = np.cos(angle),np.sin(angle)
        rx = (cx+scale*(ct*xy[0]-st*xy[1]))*rextent
        ry = (cy+scale*(st*xy[0]+ct*xy[1]))*rextent
        values = np.stack([ndi.map_coordinates(reference[:,:,c],[ry,rx],order=1,mode='nearest') for c in range(3)],axis=1)
        return values, rx, ry
    initial = np.array([proposal['cx'],proposal['cy'],np.log(proposal['scale']),proposal['angle']])
    def residual(v):
        values,rx,ry = sample(v)
        # Per-channel offsets tolerate uniform lighting, while colour variation
        # and relative arrangement remain in the residual.
        diff = values-target
        diff -= diff.mean(axis=0)
        return diff.ravel()
    lower = initial-np.array([.12,.12,math.log(1.5),np.pi/4])
    upper = initial+np.array([.12,.12,math.log(1.5),np.pi/4])
    result = least_squares(residual,initial,bounds=(lower,upper),diff_step=1e-3,
                           max_nfev=32,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    values,rx,ry = sample(result.x)
    a = values-values.mean(axis=0)
    b = target-target.mean(axis=0)
    denom = np.sqrt((a*a).sum()*(b*b).sum())
    correlation = float((a*b).sum()/denom) if denom>1e-8 else 0.
    coverage = float(((rx>=0)&(rx<rw-1)&(ry>=0)&(ry<rh-1)).mean())
    # Full query bounds, not only the sparse supporting-point hull.
    corners = np.array([[-qw,-qh],[-qw,qh],[qw,-qh],[qw,qh]],dtype=float)/(2*extent)
    cx,cy,logs,angle = result.x
    rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    transformed = corners@rot.T*np.exp(logs)+[cx,cy]
    box = [max(0.,float(transformed[:,0].min())),max(0.,float(transformed[:,1].min())),
           min(rw/rextent,float(transformed[:,0].max())),min(rh/rextent,float(transformed[:,1].max()))]
    return {'correlation':round(correlation,5),'coarse_correlation':round(proposal['correlation'],5),
            'rmse':round(float(np.sqrt(((a-b)**2).mean())),5),
            'query_contrast':round(float(np.sqrt((b*b).mean())),5),
            'source_contrast':round(float(np.sqrt((a*a).mean())),5),
            'mean_colour_error':round(float(np.sqrt(((values.mean(axis=0)-target.mean(axis=0))**2).mean())),5),
            'coverage':round(coverage,4),'samples':int(len(target)),
            'rotation_degrees':round(float(np.degrees(angle)%360),2),
            'scale':round(float(np.exp(logs)),5),'box':box,
            'transform':{'cx':float(cx),'cy':float(cy),
                         'scale':float(np.exp(logs)),'angle':float(angle)},
            'status':'Independent central-disk pixel fit; inspect source extent and alternatives'}
