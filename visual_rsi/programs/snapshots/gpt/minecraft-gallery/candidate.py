from agents.memory_agent import MemoryAgent
from visual_tools import install_visual
from patch_memory import PatchMemory, install_patches


MEMORY_GUIDANCE = """
VISUAL EVIDENCE MEMORY:
You may be asked about details after they leave view. When the scene contains
distinctive visual objects or patterns, consider STORE_PATCHES before moving on:
choose visible regions and bind EACH to its observed landmark/location and side.
Use the visible label itself, not an inferred count of steps. Capture all distinct
objects while unobstructed; a complete smaller view is often better than a large
truncated one. A broad color caption cannot identify exact patterns.
Reuse the same key for another view of the same object at the same place so
retrieval compares distinct places, not duplicate viewpoints as competing labels.
Boxes are fractions of the FULL original image: [left,top,right,bottom]. Include
the object's surface, exclude its border, held objects and surrounding scene.
Save at step 0 in PLAN when helpful; in LOG save only evidence visible NOW.
No prospective action result may be stored. Current step is the observation
step; only a previous action has an observable result.

For a later image identity question, MATCH_PATCHES retrieves numerical texture
alignment against what YOU stored. Inspect score, top_gap, source step and key.
Similar color alone or a highest rank does not establish a match. Inconclusive
scores require checking the query box or RECALL_PATCHES, not inventing a label.
RECALL_PATCHES with include_context=true shows the source landmark so you can
verify a disputed location label, then correct it explicitly with EDIT_PATCH.
Uniform query backgrounds can be trimmed when box is omitted; specify a box if
that trim includes irrelevant content. Missing memory does not prove not_seen.
Keep query images separate from witnessed world objects; storing a question as
a world sighting would contaminate later recall. EDIT_PATCH corrects labels or
deactivates mistakes. Original file and full-frame tools remain available.
"""


class Agent(MemoryAgent):
    def configure_memory(self, env_name, task, episode_idx=0):
        super().configure_memory(env_name, task, episode_idx)
        PatchMemory(self._memory_dir)

    def _get_spatial_guidance(self):
        return super()._get_spatial_guidance() + '\n' + MEMORY_GUIDANCE

    def _format_obs(self, obs):
        text = super()._format_obs(obs)
        image = getattr(self, 'current_image', None)
        if image is not None:
            text += f'\nOriginal current image dimensions: {image.width} x {image.height} pixels.'
        return text

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        context = PatchMemory(self._memory_dir).catalog()
        context += f'\nObserved now at step {self.step_count}; outcome belongs to step {self.step_count-1}.'
        return super()._log_phase(obs_text+'\n'+context, prev_action, prev_validated_action,
                                  diff_notice, spatial_guidance, map_analysis)

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice,
                             spatial_guidance, map_analysis, prev_validated_action=None):
        return super()._plan_validate_phase(obs_text+'\n'+PatchMemory(self._memory_dir).catalog(),
             instruction_prompt, diff_notice, spatial_guidance, map_analysis, prev_validated_action)


def install(agent, registry):
    install_visual(agent, registry)
    install_patches(agent, registry)
