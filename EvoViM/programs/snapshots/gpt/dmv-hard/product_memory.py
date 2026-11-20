"""LLM-selected product records and host-mediated comparison galleries."""
import json
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image
from registry import Tool
from local_features import extract, compare
from patch_match import prepare_query, search as search_patch
from regional_check import check_alignment


class ProductMemory:
    def __init__(self, agent):
        self.agent = agent
        self.root = Path(agent._memory_dir) / 'visual'
        self.path = self.root / 'products.json'
        self.last_matches = {}
        self.query_frame = None
        self.last_result = None
        self.last_find_key = None
        self.last_check = None

    def query_context(self):
        """Carry only a model-requested result from LOG into PLAN for this image."""
        if self.query_frame != self.agent.current_frame_id or not self.last_result:
            return 'No product retrieval requested for the current image yet.'
        active = self.active()
        rows = [r for r in self.last_result.get('records',[])
                if r['product_id'] in active and active[r['product_id']]['image_id']==r['image_id']]
        summary = [{'product_id':r['product_id'],'source_frame':r['frame_id'],
                    'local':r.get('match',{}),'dense':r.get('dense_match',{})} for r in rows]
        return json.dumps({'query_frame':self.query_frame,
                           'requested_filters':self.last_result.get('filters',{}),
                           'scope':self.last_result.get('scope',{}),
                           'scope_notice':self.last_result.get('scope_notice',''),
                           'searched':self.last_result['searched'], 'shortlist':summary,
                           'regional_check':self.check_context(active)},ensure_ascii=False)

    def check_context(self, active=None):
        active = self.active() if active is None else active
        result = self.last_check
        if not result or result['query_frame'] != self.agent.current_frame_id:
            return None
        # Original visual deletion may bypass product persist/invalidation.
        if any(r['product_id'] not in active or active[r['product_id']]['image_id'] != r['image_id']
               for r in result['records']):
            return None
        return result

    def invalidate_query(self):
        self.last_find_key = None
        self.last_result = None
        self.last_matches = {}
        self.last_check = None

    def records(self):
        if not self.path.exists(): return {}
        return json.loads(self.path.read_text(encoding='utf-8'))['products']

    def persist(self, rows):
        self.invalidate_query()
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'schema_version':1,'products':rows}, ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(self.path)
        # An inspectable representation keeps original file tools useful.
        self.agent.memory.write('product_catalog.txt','\n'.join(json.dumps(row,ensure_ascii=False) for row in rows.values()))

    def active(self):
        return {key: row for key,row in self.records().items()
                if row.get('active',True) and row['image_id'] in self.agent.visual.data['images']}

    def remember(self, note=''):
        page = self.agent.current_page
        if not page:
            return {'stored':False,'reason':'Current observation is not a product page. Question images are not product evidence.'}
        if self.agent.current_image is None or not self.agent.current_frame_id:
            return {'stored':False,'reason':'No current host image.'}
        rows = self.records()
        key = page['product_id']
        previous = rows.get(key)
        if previous and previous['frame_id'] == self.agent.current_frame_id and previous['image_id'] in self.agent.visual.data['images']:
            if note and note != previous['note']:
                previous['revisions'].append({'step':self.agent.step_count,'old_note':previous['note'],'reason':'LLM updated current observation description'})
                previous['note'] = note
                self.persist(rows)
            return {'stored':True,'deduplicated':True,'record':previous}
        metadata = self.agent.visual.save(self.agent.current_image,self.agent.step_count,
            f"{key} | session {page.get('session','?')} | {page.get('category','product')}"[:120],note,
            frame_id=self.agent.current_frame_id)
        features = extract(self.agent.current_image)
        # Compact descriptors keep a full catalogue checkpoint manageable;
        # positions retain float32 precision and original PNGs stay lossless.
        np.savez_compressed(self.root / f"{metadata['id']}_features.npz",
                            points=features['points'], vectors=features['vectors'].astype(np.float16))
        revisions = [] if previous is None else previous.get('revisions',[]) + [{'step':self.agent.step_count,'old_frame_id':previous['frame_id'],'old_note':previous['note'],'reason':'LLM chose to store a new observation of this product'}]
        rows[key] = {**page,'step':self.agent.step_count,'frame_id':self.agent.current_frame_id,
                     'image_id':metadata['id'],'size':list(self.agent.current_image.size),
                     'note':note,'note_status':'LLM description; check pixels','active':True,'revisions':revisions}
        self.persist(rows)
        return {'stored':True,'record':rows[key],'features':len(features['points'])}

    @staticmethod
    def normalized_text(value):
        return ' '.join(re.findall(r'\w+',str(value).casefold().replace('_',' ')))

    def find(self, query='', category='', session=None, visual=True, limit=6, strict_filters=False):
        """Explicit visual retrieval searches all selected sources unless narrowed
        deliberately. Description/category guesses are hints, not eligibility.
        Nonvisual metadata listing still applies its requested metadata filters.
        """
        all_rows = list(self.active().values())
        terms = self.normalized_text(query).split()
        def matches_hints(row):
            if category and self.normalized_text(category) not in self.normalized_text(row.get('category','')):
                return False
            if session is not None and row.get('session')!=session:
                return False
            # Search current claims and immutable provenance, never superseded notes.
            text = self.normalized_text(json.dumps({k:v for k,v in row.items() if k!='revisions'},ensure_ascii=False))
            return not terms or any(t in text for t in terms)
        hinted_rows = [r for r in all_rows if matches_hints(r)]
        restricted = bool(strict_filters or not visual)
        rows = hinted_rows if restricted else all_rows
        scope = {'mode':'metadata_filtered' if restricted else 'all_active_products',
                 'catalogue_size':len(all_rows),'eligible':len(rows),
                 'matching_hints':len(hinted_rows),'excluded':len(all_rows)-len(rows),
                 'strict_filters':bool(strict_filters)}
        if restricted:
            scope_notice = ('This search excluded products by metadata. A visual question may refer to a different category/session; choose FIND_PRODUCT with visual=true, strict_filters=false to compare all kept sources.'
                            if scope['excluded'] else
                            'Metadata filters were applied; every active product currently satisfies them.')
            if not visual:
                scope_notice += ' Metadata listing; no pixel comparison was requested.'
        else:
            scope_notice = 'The entire active catalogue is eligible. Category, session and query text are descriptive hints only; mismatching hints do not disqualify a pixel match.'
        signature = json.dumps([self.agent.current_frame_id,query,category,session,visual,limit,strict_filters,
                                [(r['product_id'],r['image_id'],r['note']) for r in all_rows]],ensure_ascii=False)
        if self.last_find_key == signature and self.last_result is not None:
            return {**self.last_result,'reused_same_query':True}
        self.last_check = None
        ranked = []
        if visual and self.agent.current_image is not None and rows:
            current = extract(self.agent.current_image)
            prepared = prepare_query(self.agent.current_image)
            for row in rows:
                path = self.root/f"{row['image_id']}_features.npz"
                if not path.exists(): continue
                with np.load(path,allow_pickle=False) as features:
                    metrics = compare(current,features)
                with Image.open(self.root/f"{row['image_id']}.png") as image:
                    dense = search_patch(prepared,image,metrics.get('transform'))
                ranked.append({**row,'match':metrics,'dense_match':dense})
            ranked.sort(key=lambda r:(r['dense_match']['correlation']*r['dense_match'].get('coverage',0),
                                      r['match']['score']),reverse=True)
            local_order = sorted(ranked,key=lambda r:(r['match']['score'],r['match']['support']),reverse=True)
            local_ranks = {r['product_id']:i+1 for i,r in enumerate(local_order)}
            for i,row in enumerate(ranked):
                row['dense_rank'] = i+1
                row['local_rank'] = local_ranks[row['product_id']]
        else:
            ranked = rows
        selected = ranked[:limit]
        if visual and self.agent.current_image is not None and len(selected)>1:
            # Independent evidence diversity without a fitted acceptance cutoff:
            # a blank query centre can hide informative boundary keypoints.
            local_leader = max(ranked,key=lambda r:(r['match']['score'],r['match']['support']))
            if local_leader not in selected:
                selected[-1] = local_leader
        results = [{**{k:v for k,v in row.items() if k!='revisions'},
                    'matches_metadata_hints':matches_hints(row)} for row in selected]
        scope['visual_comparisons'] = len(ranked) if visual and self.agent.current_image is not None else 0
        self.query_frame = self.agent.current_frame_id
        self.last_matches = {r['product_id']:{**r.get('match',{}),'dense':r.get('dense_match',{})} for r in results}
        self.last_result = {'query_frame':self.query_frame,'searched':len(rows),'returned':len(results),
                           'filters':{'query':query,'category':category,'session':session,'visual':visual},
                           'scope':scope,'scope_notice':scope_notice,
                           'records':results,
                           'interpretation':'Dense correlation checks hundreds of pixels in the query centre under one fitted scale/rotation/location, searched over the FULL source. The strongest local-feature candidate is also kept when limit>1, even if dense rank is lower: inspect disagreement. Local score and point support are separate evidence. Correlation is not a probability: near-constant/repeated patterns, partial views and distortion can be ambiguous. Low query_contrast indicates weak information, even with high correlation. Compare top-versus-alternative agreement, source extent and the gallery. Corrected note history is available via EDIT_PRODUCT, not used as current evidence.'}
        self.last_find_key = signature
        return self.last_result

    def check(self, product_ids=None, query_box=None):
        """Model-selected retrieval of fixed-transform spatial evidence.

        This cannot widen a shortlist or create/write source records. The LLM
        first chooses FIND scope and then which retrieved hypotheses to test.
        """
        if self.query_frame != self.agent.current_frame_id or self.agent.current_image is None:
            return {'checked':False,'reason':'First request visual FIND_PRODUCT for the CURRENT image.'}
        if query_box is not None:
            valid_box = (len(query_box)==4 and all(isinstance(v,(float,int)) and
                         not isinstance(v,bool) and math.isfinite(v) for v in query_box))
            if not valid_box or not (0<=query_box[0]<query_box[2]<=1 and 0<=query_box[1]<query_box[3]<=1):
                return {'checked':False,'reason':'query_box must be [left,top,right,bottom] in [0,1], with positive width/height.'}
        keys = list(self.last_matches) if product_ids is None else list(dict.fromkeys(product_ids))
        if not keys:
            return {'checked':False,'reason':'No sources selected. First FIND_PRODUCT, then choose its product ids.'}
        rows = self.active()
        if any(key not in rows or key not in self.last_matches for key in keys):
            return {'checked':False,'reason':'Every chosen id must be active and in the CURRENT FIND_PRODUCT shortlist. Widen FIND_PRODUCT if needed.'}
        results = []
        for key in keys:
            row, match = rows[key], self.last_matches[key]
            hypotheses = [('dense_centre',match.get('dense',{}).get('transform')),
                          ('local_geometry',match.get('transform'))]
            checked = []
            with Image.open(self.root/f"{row['image_id']}.png") as image:
                for name, transform in hypotheses:
                    if transform is not None:
                        checked.append({'hypothesis':name,
                                        **check_alignment(self.agent.current_image,image,transform,query_box)})
            if not checked:
                return {'checked':False,'reason':'Shortlist has no current pixel alignment. Request FIND_PRODUCT with visual=true.'}
            results.append({'product_id':key,'url':row['url'],'image_id':row['image_id'],
                            'source_frame':row['frame_id'],'hypotheses':checked})
        self.last_check = {'checked':True,'query_frame':self.agent.current_frame_id,
                           'query_box':query_box or [0.,0.,1.,1.],
                           'retrieval_scope':self.last_result.get('scope',{}) if self.last_result else {},
                           'checked_sources':len(results),'shortlist_size':len(self.last_matches),
                           'selection':'Only LLM-requested current shortlist sources; no automatic reranking.',
                           'records':results,
                           'interpretation':'Use outside_centre and cell evidence to test whether a central fit extends to informative outer clues. Transform and central colour offset are fixed before checking outer pixels. Low-information cells, query padding, occlusion and perspective can disagree for a correct source. Inspect a gallery; do not treat these metrics as a verdict.'}
        return self.last_check

    def edit(self, product_id='', note=None, active=None, reason=''):
        key = product_id or (self.agent.current_page or {}).get('product_id')
        rows = self.records()
        if key not in rows: return {'updated':False,'reason':'No matching stored product; use an id from FIND_PRODUCT.'}
        row = rows[key]
        if note is None and active is None: return row
        row['revisions'].append({'step':self.agent.step_count,'old_note':row['note'],'old_active':row['active'],'reason':reason})
        if note is not None: row['note'] = note
        if active is not None: row['active'] = active
        self.persist(rows)
        return {'updated':True,'record':row}

    def gallery(self, product_ids=None, matched_regions=False, include_current=True):
        rows = self.active()
        # {} is an explicit request for the previous tool's shortlist. Without
        # a shortlist return metadata, never choose new historical frames.
        keys = list(self.last_matches) if product_ids is None else product_ids
        if product_ids is None and self.query_frame != self.agent.current_frame_id:
            return {'requested':False,'reason':'Last shortlist belongs to another query image. Request FIND_PRODUCT for the CURRENT image, or explicitly choose product_ids.'}
        selected = [rows[k] for k in keys if k in rows][:8]
        if not selected: return {'requested':False,'reason':'First FIND_PRODUCT, or specify stored product_ids.'}
        ops, sources, labels = [], [], []
        if include_current and self.agent.current_frame_id:
            ops.extend([{'op':'frame','id':self.agent.current_frame_id,'as':'query'},
                        {'op':'resize','src':'query','max_side':384,'as':'query_small'}])
            sources.append('query_small'); labels.append('CURRENT QUERY')
        for i,row in enumerate(selected):
            src = f'frame{i}'
            ops.append({'op':'frame','id':row['frame_id'],'as':src})
            match = self.last_matches.get(row['product_id'],{}) if self.query_frame==self.agent.current_frame_id else {}
            dense = match.get('dense',{})
            box = (dense.get('box') or match.get('box')) if matched_regions else None
            if box and (dense.get('box') or match.get('support',0)>=2):
                w,h = row['size']
                extent = max(w,h)
                x0,y0,x1,y1 = box
                pad = max(x1-x0,y1-y0)*.05 + .01
                pixel_box = [max(0,math.floor((x0-pad)*extent)),max(0,math.floor((y0-pad)*extent)),
                             min(w,math.ceil((x1+pad)*extent)),min(h,math.ceil((y1+pad)*extent))]
                if pixel_box[2]>pixel_box[0] and pixel_box[3]>pixel_box[1]:
                    cropped = f'crop{i}'
                    ops.append({'op':'crop','src':src,'box':pixel_box,'as':cropped})
                    src = cropped
            small = f'small{i}'
            ops.append({'op':'resize','src':src,'max_side':384,'as':small})
            sources.append(small)
            labels.append(f"{row['product_id']} S{row.get('session','?')} P{row.get('page','?')}")
        ops.append({'op':'tile','srcs':sources,'cols':min(3,len(sources)),'labels':labels,'as':'out'})
        self.agent.visual_request = {'ops':ops,'output':'out'}
        return {'requested':True,'products':[{'product_id':r['product_id'],'image_id':r['image_id']} for r in selected],
                'query_frame':self.agent.current_frame_id,
                'comparison':[{'product_id':r['product_id'],'dense':self.last_matches.get(r['product_id'],{}).get('dense',{})} for r in selected] if self.query_frame==self.agent.current_frame_id else [],
                'view':'matched regions' if matched_regions else 'full product frames','next':'Inspect the historical gallery in the next model request.'}


def specs():
    string = {'type':'string','maxLength':1000}
    obj = {'type':'object','additionalProperties':False,'required':[]}
    return [
        {'name':'REMEMBER_PRODUCT','description':'Choose to store the CURRENT observed product image and exact URL together. Automatically extracts only public page identity and local pixel features; you supply a distinguishing note. Safe at step 0 in PLAN. Idempotent for the same frame; never stores a predicted action outcome.',
         'parameters':{**obj,'properties':{'note':string}},'phases':['LOG','PLAN'],'permission':'memory_write','returns':'metadata','handler':'remember'},
        {'name':'FIND_PRODUCT','description':'Retrieve products you chose to store. visual=true (default) searches the ENTIRE active catalogue; category/session/query are hints and never exclude pixel candidates unless you explicitly set strict_filters=true. Use strict filtering only to inspect a known subset. visual=false lists metadata with the requested filters. Returns search scope/counts, dense scale/rotation/location agreement, local support and their separate ranks; retains the local leader when limit>1. Results are scoped to THIS image. Request PRODUCT_GALLERY before answering; notes and correlations are not correctness labels.',
         'parameters':{**obj,'properties':{'query':string,'category':string,'session':{'type':'integer','minimum':1},'visual':{'type':'boolean'},'limit':{'type':'integer','minimum':1,'maximum':8},'strict_filters':{'type':'boolean','description':'Explicitly exclude products outside category/session/text filters during visual retrieval. Default false treats them as hints and searches all kept sources.'}}},'phases':['LOG','PLAN'],'permission':'memory_read','returns':'metadata','handler':'find'},
        {'name':'PRODUCT_GALLERY','description':'Compare stored product frames labelled with exact URL ids through the host image channel. With {} uses your last FIND_PRODUCT shortlist for THIS query; specify product_ids to choose another set. matched_regions=true shows the fitted FULL query extent in each source, with rotation/scale and dense correlation in text. It avoids tiny two-point crops. Inspect the next request and compare spatial arrangement before answering.',
         'parameters':{**obj,'properties':{'product_ids':{'type':'array','maxItems':8,'items':{'type':'string'}},'matched_regions':{'type':'boolean'},'include_current':{'type':'boolean'}}},'phases':['LOG','PLAN'],'permission':'memory_read','returns':'image','handler':'gallery'},
        {'name':'CHECK_PRODUCT','description':'Optional after visual FIND_PRODUCT: check its fixed dense-centre and local-geometry alignments against pixels across the current query, including OUTSIDE the fitted centre. Returns coverage, contrast, correlation and centre-calibrated error per region and a 3x3 grid. Use when central evidence is ambiguous or outer clues need checking. Does not change the shortlist or choose an answer. {} checks the current shortlist; product_ids explicitly selects ids from it. Optional query_box=[left,top,right,bottom] in normalized current-image coordinates restricts reported evidence to a region you choose. No refitting, no new source writes, no image delivery. Query padding or blank cells are not a veto.',
         'parameters':{**obj,'properties':{'product_ids':{'type':'array','maxItems':8,'items':{'type':'string'}},'query_box':{'type':'array','maxItems':4,'items':{'type':'number','minimum':0,'maximum':1}}}},'phases':['LOG','PLAN'],'permission':'memory_read','returns':'metadata','handler':'check'},
        {'name':'EDIT_PRODUCT','description':'Correct a stored description or deactivate/reactivate a record, with a reason. Source URL and frame stay immutable. Descriptions remain LLM claims. With no id uses the current product, if stored. With no changes reads a record.',
         'parameters':{**obj,'properties':{'product_id':string,'note':string,'active':{'type':'boolean'},'reason':string}},'phases':['LOG','PLAN'],'permission':'memory_write','returns':'metadata','handler':'edit'}]


def install_products(agent, registry):
    agent.products = ProductMemory(agent)
    for spec in specs():
        registry.register(Tool(spec['name'],spec['description'],spec['parameters'],
                          getattr(agent.products,spec['handler']),tuple(spec['phases']),spec['permission'],spec['returns']))
