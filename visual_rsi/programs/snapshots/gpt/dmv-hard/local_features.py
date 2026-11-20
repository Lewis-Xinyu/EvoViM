"""Bounded local image descriptors; no semantic labels or episode constants.

Harris corners across an image pyramid, oriented gradient histograms and
spatial colour samples. Scores are retrieval aids, never world actions.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from PIL import Image


def extract(image):
    image = image.convert('RGB')
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    points, vectors = [], []
    side = max(image.size)
    while side >= 48:
        size = (max(1, round(image.width * side / max(image.size))),
                max(1, round(image.height * side / max(image.size))))
        rgb = np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        gy = ndi.gaussian_filter(gray, 1, order=(1, 0))
        gx = ndi.gaussian_filter(gray, 1, order=(0, 1))
        a = ndi.gaussian_filter(gx * gx, 2)
        b = ndi.gaussian_filter(gy * gy, 2)
        c = ndi.gaussian_filter(gx * gy, 2)
        response = a * b - c * c - .04 * (a + b) ** 2
        maxima = (response == ndi.maximum_filter(response, size=7)) & (response > 0)
        maxima[:12] = False
        maxima[-12:] = False
        maxima[:, :12] = False
        maxima[:, -12:] = False
        ys, xs = np.where(maxima)
        selected = np.argsort(response[ys, xs])[::-1][:48]
        oy, ox = np.mgrid[-8:9, -8:9].astype(np.float32)
        weights = np.exp(-(ox * ox + oy * oy) / 32)
        sy, sx = np.mgrid[-16:16, -16:16].astype(np.float32) + .5
        patch_weights = np.exp(-(sx * sx + sy * sy) / 128)
        cell = (np.arange(32) // 8)[:, None] * 4 + (np.arange(32) // 8)[None, :]
        for k in selected:
            y, x = ys[k], xs[k]
            dx = gx[y-8:y+9, x-8:x+9]
            dy = gy[y-8:y+9, x-8:x+9]
            angles = np.arctan2(dy, dx)
            bins = np.floor((angles + np.pi) * 36 / (2 * np.pi)).astype(int) % 36
            hist = np.bincount(bins.ravel(), weights=(np.hypot(dx, dy)*weights).ravel(), minlength=36)
            hist = ndi.gaussian_filter1d(hist, 1, mode='wrap')
            peaks = np.where((hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1)))[0]
            peaks = sorted(peaks, key=lambda j: hist[j], reverse=True)[:2]
            for j in peaks:
                if hist[j] < .8 * hist.max():
                    continue
                theta = (j + .5) * (2 * np.pi / 36) - np.pi
                ct, st = np.cos(theta), np.sin(theta)
                px = x + ct * sx - st * sy
                py = y + st * sx + ct * sy
                dx = ndi.map_coordinates(gx, [py, px], order=1, mode='reflect')
                dy = ndi.map_coordinates(gy, [py, px], order=1, mode='reflect')
                orient = (np.arctan2(dy, dx) - theta) % (2 * np.pi) * 8 / (2 * np.pi)
                lo = np.floor(orient).astype(int) % 8
                frac = orient - np.floor(orient)
                mag = np.hypot(dx, dy) * patch_weights
                desc = np.bincount((cell * 8 + lo).ravel(), weights=(mag*(1-frac)).ravel(), minlength=128)
                desc += np.bincount((cell * 8 + (lo+1)%8).ravel(), weights=(mag*frac).ravel(), minlength=128)
                desc /= max(np.linalg.norm(desc), 1e-8)
                desc = np.minimum(desc, .2)
                desc /= max(desc.sum(), 1e-8)
                desc = np.sqrt(desc)
                # Colour is centred to tolerate a uniform lighting offset.
                colour = np.stack([ndi.map_coordinates(rgb[:,:,ch], [py,px], order=1, mode='reflect') for ch in range(3)],axis=2)
                colour = colour.reshape(4,8,4,8,3).mean(axis=(1,3)).reshape(-1)
                colour -= colour.mean()
                colour /= max(np.linalg.norm(colour), 1e-8)
                vector = np.concatenate([desc, .35*colour])
                vector /= max(np.linalg.norm(vector), 1e-8)
                points.append([x / max(size), y / max(size), 1 / max(size), theta])
                vectors.append(vector)
        side = int(side / np.sqrt(2))
    return {'centre': np.asarray(image.size,dtype=float)/(2*max(image.size)),
            'points': np.asarray(points, dtype=np.float32).reshape(-1,4),
            'vectors': np.asarray(vectors, dtype=np.float32).reshape(-1,176)}


def compare(query, reference):
    q, r = query['vectors'], reference['vectors']
    if len(q) < 2 or len(r) < 2:
        return {'score':0., 'support':0, 'mean_distance':None, 'box':None}
    distances = cdist(q, r, 'sqeuclidean')
    nearest = np.argmin(distances, axis=1)
    best = distances[np.arange(len(q)),nearest]
    second = np.partition(distances,1,axis=1)[:,1]
    # Standard descriptor distinctiveness ratio; not a product acceptance rule.
    good = np.where(best < .8**2 * np.maximum(second, 1e-9))[0]
    good = sorted(good, key=lambda k:best[k])[:96]
    if not good:
        return {'score':0., 'support':0, 'mean_distance':round(float(np.sqrt(best.mean())),4), 'box':None}
    qp = query['points'][good]
    rp = reference['points'][nearest[good]]
    weights = np.maximum(0, 1 - np.sqrt(best[good])/np.sqrt(2))
    # Each correspondence proposes a similarity transform from its scale and
    # orientation. Count spatially consistent DISTINCT query/reference points.
    winner, winner_score, winner_transform = None, -1., None
    for i in range(len(good)):
        scale = rp[i,2] / qp[i,2]
        theta = rp[i,3] - qp[i,3]
        rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        predicted = (qp[:,:2] - qp[i,:2]) @ rot.T * scale + rp[i,:2]
        error = np.linalg.norm(predicted-rp[:,:2],axis=1)
        tolerance = 6 * np.maximum(rp[:,2], qp[:,2]*scale)
        coherent = error <= tolerance
        picked, seenq, seenr = [], set(), set()
        for k in np.argsort(weights)[::-1]:
            if not coherent[k]: continue
            qkey = tuple(np.round(qp[k,:2],2))
            rkey = tuple(np.round(rp[k,:2],2))
            if qkey in seenq or rkey in seenr: continue
            seenq.add(qkey); seenr.add(rkey); picked.append(k)
        score = sum(float(weights[k]) for k in picked)
        if score > winner_score:
            winner, winner_score = picked, score
            centre = np.asarray(query.get('centre',[.5,.5]))
            mapped_centre = (centre-qp[i,:2]) @ rot.T*scale+rp[i,:2]
            winner_transform = {'cx':float(mapped_centre[0]),'cy':float(mapped_centre[1]),
                                'scale':float(scale),'angle':float(theta)}
    region = rp[winner,:2]
    box = [float(region[:,0].min()),float(region[:,1].min()),float(region[:,0].max()),float(region[:,1].max())]
    return {'score':round(winner_score,4), 'support':len(winner),
            'mean_distance':round(float(np.sqrt(best[np.asarray(good)[winner]]).mean()),4),
            'box':box, 'transform':winner_transform}
