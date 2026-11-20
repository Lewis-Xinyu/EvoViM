"""Observation-bound product memory; every persistent write is model chosen."""
import re
from agents import memory_agent as legacy
from prompt_builder.history import Message
from visual_tools import install_visual
from product_memory import install_products


class Agent(legacy.MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory,prompt_builder,config)
        self.current_page = None

    def _get_initial_files(self):
        files = super()._get_initial_files()
        files['product_catalog.txt'] = ''
        return files

    def _format_obs(self, obs):
        text = super()._format_obs(obs)
        self.current_page = None
        # Read identity only from a product-page observation, never the URL
        # placeholder in a question or an earlier answer.
        page = re.search(r'Product page\s+(\d+)(?:/(\d+))?:\s*URL\s+(/product/([^\s.]+))',text,re.I)
        if page:
            session = re.search(r'Session\s+(\d+)\s*/\s*\d+',text,re.I)
            category = re.search(r"browsing category\s+['\"]([^'\"]+)['\"]",text,re.I)
            self.current_page = {'product_id':page[4],'url':page[3],'page':int(page[1]),
                                 'session':int(session[1]) if session else None,
                                 'category':category[1] if category else ''}
        return text

    def _memory_advice(self):
        return ("PRODUCT MEMORY: Keep each observed product's exact URL bound to its image. "
                "REMEMBER_PRODUCT stores the current page, image and your note in one explicit call. "
                "Capture distinctive material, geometry, decoration, small patches and other image regions; "
                "generic object colour often cannot identify an exact product. Notes are fallible; keep pixels. "
                "FIND_PRODUCT compares local visual features across your stored products and returns a shortlist. "
                "Visual retrieval searches your entire active catalogue, even if you supply category/session/text hints. "
                "Those hints describe a guess; only strict_filters=true deliberately excludes other products. "
                "It also independently fits the query centre to retained source pixels over scale, rotation "
                "and location. Compare dense correlation, coverage, source extent and alternatives as well "
                "as local support. A high dense correlation over hundreds of pixels is stronger evidence "
                "than a generic similarity in a fallible note; ambiguous or repeated patches still need inspection. "
                "Low query contrast makes even a high correlation weak; the shortlist also retains the "
                "strongest local candidate so informative edges are not lost when the query centre is blank. "
                "PRODUCT_GALLERY shows that shortlist labelled with exact product ids; matched_regions=true "
                "isolates the fitted full-query extent rather than only a few supporting points. Verify pixels, including spatial arrangement and decoration, "
                "before choosing an id. When centre evidence is ambiguous or outer clues matter, "
                "CHECK_PRODUCT optionally checks those retrieved transforms across the query and reports "
                "outside-centre and grid-cell support. Blank cells or query padding are not a veto; "
                "compare informative regions and alternatives. You may correct or deactivate descriptions with EDIT_PRODUCT. "
                "All original text and image tools remain available. Choose what to store and retrieve. "
                "Issue exactly one command per reply, then inspect its actual result.")

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        effective = prev_validated_action or prev_action
        prompt = (f"LOG at step {self.step_count}: the previous action at step {self.step_count-1} "
                  f"was {effective!r}. Its observed response is below. Attribute any outcome to that "
                  "previous action; do not claim an outcome for the action you will choose next. "
                  "A newly displayed product is a CURRENT observation, not a description of the previous image. "
                  "If this response displays a new product, you can preserve it now with REMEMBER_PRODUCT "
                  "and a distinguishing note. If it displays a question, preserve the distinction between "
                  "the previous answer and the new question. 'Answer recorded' does not mean correct. "
                  "Do not store a query image as a new product. When finished use <|DONE|>.\n\n"
                  + self._memory_advice()+f"\n\nOBSERVED RESPONSE:\n{obs_text}\n\nMEMORY FILES:\n{self.memory.file_index()}")
        return self._run_memory_phase('LOG',prompt)

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice,
                             spatial_guidance, map_analysis, prev_validated_action=None):
        if self.current_page:
            direction = ("The observation is a product page. Decide whether to preserve it using "
                         "REMEMBER_PRODUCT before moving on, including the first page at step 0. "
                         "Then follow the current page's instruction. Repeated storage of the same "
                         "frame is deduplicated. Product storage records an observation, never the "
                         "future outcome of your next action.")
        else:
            direction = ("Read the CURRENT question and its image. For an exact-product visual question, "
                         "use FIND_PRODUCT across all remembered sessions; the category browsed most recently "
                         "may be irrelevant. Request PRODUCT_GALLERY and inspect the candidates before "
                         "choosing. Enlarged or rotated patches need local arrangement comparisons; numeric "
                         "scores are aids, not correctness labels. If LOG already requested retrieval for this "
                         "same query, its numeric evidence is below; you can request its gallery directly. "
                         "Check the retrieval scope: all_active_products means every kept source was eligible. "
                         "If it says metadata_filtered and excludes products, request FIND_PRODUCT with "
                         "strict_filters=false and visual=true before excluding other products on a category guess. "
                         "If necessary list another category or "
                         "session (visual=false) and request its gallery. Follow the question's answer "
                         "format exactly, using an id copied from the retrieved record. Previous answers "
                         "are not verified facts about this new question.")
        query_context = self.products.query_context() if hasattr(self,'products') else ''
        prompt = (instruction_prompt+'\n\n'+self._memory_advice()+'\n\n'+direction+
                  f"\n\nCURRENT OBSERVATION:\n{obs_text}\n\nMEMORY FILES:\n{self.memory.file_index()}\n\n"
                  f"EXPLICITLY RETRIEVED EVIDENCE FOR CURRENT IMAGE (if any):\n{query_context}\n\n"
                  "Use the host's ACTION command to commit exactly one world action when ready.")
        return self._run_memory_phase('PLAN',prompt)

    def _run_memory_phase(self, phase, prompt):
        messages = [Message(role='user',content=prompt)]
        turns = []
        total_in = total_out = 0
        final_action = None
        model_id = 'unknown'
        count = self.max_log_iterations if phase=='LOG' else self.max_validate_iterations
        for iteration in range(count):
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                turns.append({'prompt':messages[-1].content,'raw_response':f'[ERROR: {exc}]','tool_results':''})
                break
            total_in += response.input_tokens
            total_out += response.output_tokens
            model_id = response.model_id
            text = response.completion
            # Resolve the original module functions at call time: the host owns
            # decoding, phase restrictions and the registry dispatch.
            calls, finished = (legacy.parse_log_calls(text) if phase=='LOG' else legacy.parse_plan_calls(text))
            results = []
            for op,args in calls:
                result = (self._execute_write_op(op,args) if phase=='LOG' else self._execute_read_op(op,args))
                _,formatted = self._format_tool_result(op,args,result)
                results.append(formatted)
            result_text = '\n\n'.join(results)
            turns.append({'prompt':messages[-1].content,'raw_response':text,'tool_results':result_text,
                          'committed_action':finished if phase=='PLAN' else None})
            messages.append(Message(role='assistant',content=text))
            if phase=='LOG' and finished: break
            if phase=='PLAN' and finished is not None:
                final_action = finished
                break
            if not calls: break
            remaining = count-iteration-1
            continuation = ('Finish logging with <|DONE|> when complete.' if phase=='LOG' else
                            'Inspect retrieved pixels and facts, then follow the CURRENT question or page instruction. Commit with ACTION when ready.')
            if remaining<=2:
                continuation += ' Few requests remain; finish this phase after reviewing the available evidence.'
            messages.append(Message(role='user',content=f"ACTUAL TOOL RESULTS:\n{result_text}\n\n{continuation}"))
        debug = {'initial_prompt':prompt,'turns':turns,'input_tokens':total_in,'output_tokens':total_out}
        if phase=='PLAN': debug.update(final_action=final_action if final_action is not None else self._get_fallback_action(),model_id=model_id)
        return debug

    def export_state(self):
        return {'step_count':self.step_count,'current_page':self.current_page,
                '_prev_obs_text':self._prev_obs_text,'_unchanged_count':self._unchanged_count,
                'last_step_debug':self.last_step_debug,
                'last_matches':self.products.last_matches if hasattr(self,'products') else {},
                'query_frame':self.products.query_frame if hasattr(self,'products') else None,
                'last_result':self.products.last_result if hasattr(self,'products') else None,
                'last_find_key':self.products.last_find_key if hasattr(self,'products') else None,
                'last_check':self.products.last_check if hasattr(self,'products') else None}

    def import_state(self, state):
        for key in ('step_count','current_page','_prev_obs_text','_unchanged_count','last_step_debug'):
            if key in state: setattr(self,key,state[key])
        if hasattr(self,'products'):
            self.products.last_matches = state.get('last_matches',{})
            self.products.query_frame = state.get('query_frame')
            self.products.last_result = state.get('last_result')
            self.products.last_find_key = state.get('last_find_key')
            self.products.last_check = state.get('last_check')

    def reset(self):
        super().reset()
        self.current_page = None
        if hasattr(self,'visual') and self.visual is not None:
            self.visual.root.mkdir(parents=True,exist_ok=True)
            self.visual.data = {'schema_version':1,'next_id':1,'images':{}}
            self.visual._persist()
        if hasattr(self,'products'):
            self.products.invalidate_query()
            self.products.query_frame = None


def install(agent, registry):
    install_visual(agent,registry)
    install_products(agent,registry)
