# Interactive World Model: From Navigation to Object Interaction
# 交互式世界模型研究报告：从导航到物体交互

> **Topic**: Interactive World Model with Object-Level Interactions
> **Base Codebase**: YUME 1.5
> **Date**: 2026-02-18

---

## Table of Contents

- [1. Background & Gap Analysis](#1-background--gap-analysis)
- [2. Innovation Direction 1: Extended Interaction Action Space](#2-innovation-direction-1-extended-interaction-action-space)
- [3. Innovation Direction 2: Object State Memory](#3-innovation-direction-2-object-state-memory)
- [4. Innovation Direction 3: Hierarchical Interaction Generation](#4-innovation-direction-3-hierarchical-interaction-generation)
- [5. Innovation Direction 4: Interaction-Aware RL Post-Training](#5-innovation-direction-4-interaction-aware-rl-post-training)
- [6. Innovation Direction 5: Affordance-Conditioned Generation](#6-innovation-direction-5-affordance-conditioned-generation)
- [7. Data Engine: Interaction Data Pipeline](#7-data-engine-interaction-data-pipeline)
- [8. Training Recipe & Roadmap](#8-training-recipe--roadmap)

---

## 1. Background & Gap Analysis

### 1.1 YUME1.5 Core Architecture

| Component | Detail |
|-----------|--------|
| Base Model | Wan2.2-5B DiT (32 layers, hidden_dim=2048, ffn_dim=8192) |
| Training | Flow Matching loss + Self-Forcing + DMD distillation |
| Action Encoding | Discrete text: WASD (movement) + arrow keys (camera), injected via T5 text encoder |
| Memory | TSCM: temporal-spatial compression (Patchify interpolation) + channel compression (linear attention) |
| Speed | ~12fps @ 540p, 4 inference steps, single A100 |
| Data Format | MP4 + TXT (Keys/Mouse) + NPY (camera c2w poses) |

### 1.2 HY-WorldPlay Core Architecture

| Component | Detail |
|-----------|--------|
| Base Model | HunyuanVideo-1.5 DiT |
| Action Encoding | **Dual**: discrete keyboard (MLP → timestep embedding) + continuous camera pose (PRoPE in self-attention) |
| Memory | Reconstituted Context Memory: FOV overlap retrieval + Temporal Reframing (rewrite RoPE indices) |
| RL Post-Training | WorldCompass: clip-level rollout + action following reward + visual quality reward |
| Distillation | Context Forcing: align memory context between teacher & student |
| Speed | ~24fps, 8×GPU with sequence parallelism |
| Data | 320K clips: AAA games (53%) + DL3DV (19%) + UE rendered (16%) + Sekai (12%) |

### 1.3 Critical Gap: No Object Interaction

**Both YUME1.5 and HY-WorldPlay only model navigation (moving through a world), not interaction (changing the world).**

| Capability | YUME1.5 | HY-WorldPlay | Ours (Target) |
|-----------|---------|--------------|---------------|
| Camera movement | ✅ | ✅ | ✅ |
| Person movement | ✅ | ✅ | ✅ |
| Text-triggered events | ✅ (basic) | ✅ (basic) | ✅ |
| **Object pickup/place** | ❌ | ❌ | **✅** |
| **Object state change** | ❌ | ❌ | **✅** |
| **State persistence on revisit** | ❌ | ❌ | **✅** |
| **Physical causality** | ❌ | ❌ | **✅** |

### 1.4 HY-WorldPlay Strengths to Borrow

1. **Dual Action Representation**: We should support both discrete and continuous control for interaction.
2. **WorldCompass RL**: The RL post-training framework can be adapted for interaction-specific rewards.
3. **Context Forcing**: Their distillation method is strictly better than YUME's Self-Forcing + DMD for memory-aware models.

### 1.5 HY-WorldPlay Limitations We Address

1. Their memory is purely **geometric** (FOV overlap) — not semantic/object-aware.
2. Their "promptable events" are open-loop text injection — no causal object state tracking.
3. No structured interaction data — their AAA game data has interactions but without annotation.

---

## 2. Innovation Direction 1: Extended Interaction Action Space

### 2.1 Motivation

YUME1.5's action space is purely navigational:

```python
# Current YUME1.5 action vocabulary (fastvideo/dataset/t2v_datasets.py:397-422)
vocab_movement = {
    "W": "Person moves forward (W).",
    "A": "Person moves left (A).",
    # ... 9 total movement actions
}
vocab_camera = {
    "→": "Camera turns right (→).",
    "←": "Camera turns left (←).",
    # ... 9 total camera actions
}
```

We need to add **interaction actions** that operate on objects:

### 2.2 Design: Tri-Part Action Description

```python
# === NEW: Extended Action Vocabulary ===

vocab_movement = {  # Unchanged from YUME1.5
    "W": "Person moves forward (W).",
    "A": "Person moves left (A).",
    "S": "Person moves backward (S).",
    "D": "Person moves right (D).",
    "W+A": "Person moves forward and left (W+A).",
    "W+D": "Person moves forward and right (W+D).",
    "S+D": "Person moves backward and right (S+D).",
    "S+A": "Person moves backward and left (S+A).",
    "None": "Person stands still (·).",
}

vocab_camera = {  # Unchanged from YUME1.5
    "→": "Camera turns right (→).",
    "←": "Camera turns left (←).",
    "↑": "Camera tilts up (↑).",
    "↓": "Camera tilts down (↓).",
    "↑→": "Camera tilts up and turns right (↑→).",
    "↑←": "Camera tilts up and turns left (↑←).",
    "↓→": "Camera tilts down and turns right (↓→).",
    "↓←": "Camera tilts down and turns left (↓←).",
    "·": "Camera remains still (·).",
}

# === NEW ACTION VOCABULARY ===
vocab_interaction = {
    "pickup": "Agent picks up {object}.",
    "place": "Agent places {object} on {target}.",
    "drop": "Agent drops {object}.",
    "throw": "Agent throws {object}.",
    "push": "Agent pushes {object}.",
    "open": "Agent opens {object}.",
    "close": "Agent closes {object}.",
    "toggle_on": "Agent turns on {object}.",
    "toggle_off": "Agent turns off {object}.",
    "slice": "Agent slices {object}.",
    "break": "Agent breaks {object}.",
    "hit": "Agent hits {object}.",
    "pet": "Agent pets {animal}.",
    "none": "No interaction.",
}
```

### 2.3 Implementation in YUME1.5 Codebase

#### File: `fastvideo/dataset/t2v_datasets.py`

**Modification to `StableVideoAnimationDataset.get_sample()`** (line ~397-436):

```python
def get_sample(self, index):
    video_path, videoid, keys, mouse, interaction, target_obj, \
        npz_path, start_frame, end_frame, flag, full_mp4_1 = self.vid_meta[index]

    caption = "This video depicts a first-person interactive scene."

    # 1) Movement description (unchanged)
    caption += vocab_movement[keys[0]]

    # 2) Camera description (unchanged)
    caption += vocab_camera[mouse[0]]

    # 3) NEW: Interaction description
    if interaction[0] != "none":
        interact_desc = vocab_interaction[interaction[0]]
        interact_desc = interact_desc.format(
            object=target_obj[0] if target_obj[0] else "the object",
            target=target_obj[1] if len(target_obj) > 1 and target_obj[1] else "the surface",
            animal=target_obj[0] if target_obj[0] else "the animal",
        )
        caption += interact_desc
    else:
        caption += vocab_interaction["none"]

    # 4) Optional: camera trajectory metrics (unchanged from YUME1.5)
    if npz_path is not None:
        # ... existing metric computation ...
        pass

    return sample
```

#### Why This Works Without Architecture Changes

YUME1.5's key insight is that action descriptions are part of the **text prompt** fed to the T5 encoder. Since the set of interaction verbs is finite (~15), all interaction descriptions can be **pre-computed and cached** as T5 embeddings — exactly like YUME1.5 does for movement/camera descriptions (Section 4.1 of the paper). This means:

- **Zero additional inference cost** for interaction encoding
- **No model architecture changes** needed
- Interaction semantics are learned through the existing cross-attention mechanism

### 2.4 Data Format for Interaction Actions

Each training sample's TXT annotation file extends from:

```
# Current YUME1.5 format:
Keys: W
Mouse: →
Start Frame: 0
End Frame: 128
```

To:

```
# Extended format:
Keys: W
Mouse: →
Interaction: pickup
Target Object: Apple
Target Receptacle:
Object State Before: on_table
Object State After: in_hand
Start Frame: 0
End Frame: 128
```

#### Corresponding parser modification:

```python
def parse_txt_file_extended(txt_path):
    """Parse extended TXT file with interaction annotations."""
    keys = mouse = interaction = target_obj = target_recep = None
    state_before = state_after = None
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('Keys:'):
                    keys = line.split(':', 1)[1].strip()
                elif line.startswith('Mouse:'):
                    mouse = line.split(':', 1)[1].strip()
                elif line.startswith('Interaction:'):
                    interaction = line.split(':', 1)[1].strip()
                elif line.startswith('Target Object:'):
                    target_obj = line.split(':', 1)[1].strip()
                elif line.startswith('Target Receptacle:'):
                    target_recep = line.split(':', 1)[1].strip()
                elif line.startswith('Object State Before:'):
                    state_before = line.split(':', 1)[1].strip()
                elif line.startswith('Object State After:'):
                    state_after = line.split(':', 1)[1].strip()
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
    return keys, mouse, interaction or "none", target_obj or "", \
           target_recep or "", state_before or "", state_after or ""
```

---

## 3. Innovation Direction 2: Object State Memory

### 3.1 Motivation

HY-WorldPlay's Reconstituted Context Memory is purely **geometric**: it retrieves past frames based on camera FOV overlap. It answers "what did this location look like?" but cannot answer "what is the state of this object?"

Example failure: Agent picks up an apple at time t=10, walks away, walks back at t=100. The geometric memory retrieves the frame from t=10, which shows the apple still on the table. The model generates the apple as still there — violating object permanence.

### 3.2 Design: Object State Graph (OSG)

```
┌─────────────────────────────────────────────────────────┐
│                   Object State Graph                     │
│                                                          │
│  Object_001 (Apple)                                      │
│    ├─ location: table_kitchen → agent_hand → table_living│
│    ├─ state: intact                                      │
│    └─ last_interaction: t=50, action=place               │
│                                                          │
│  Object_002 (Door)                                       │
│    ├─ state: closed → open                               │
│    └─ last_interaction: t=30, action=open                │
│                                                          │
│  Object_003 (Lamp)                                       │
│    ├─ state: off → on                                    │
│    └─ last_interaction: t=45, action=toggle_on           │
│                                                          │
│  Object_004 (Tree)                                       │
│    ├─ state: intact → chopped                            │
│    └─ last_interaction: t=70, action=hit                 │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Implementation

#### New file: `fastvideo/memory/object_state_memory.py`

```python
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ObjectState:
    object_id: str
    object_type: str
    state: str              # "intact", "open", "in_hand", "chopped", "on", "off", ...
    location: str           # "table_kitchen", "agent_hand", "floor", ...
    last_interact_time: int # chunk index when last interacted
    last_action: str        # "pickup", "place", "open", etc.
    bbox: Optional[Tuple[float, float, float, float]] = None


class ObjectStateGraph:
    """Tracks object states across autoregressive generation chunks."""

    def __init__(self, max_objects: int = 50):
        self.max_objects = max_objects
        self.objects: Dict[str, ObjectState] = {}
        self.history: List[Dict] = []

    def update(self, object_id: str, object_type: str,
               action: str, new_state: str, new_location: str,
               chunk_idx: int, bbox: Optional[Tuple] = None):
        """Update object state after an interaction."""
        self.objects[object_id] = ObjectState(
            object_id=object_id,
            object_type=object_type,
            state=new_state,
            location=new_location,
            last_interact_time=chunk_idx,
            last_action=action,
            bbox=bbox,
        )
        self.history.append({
            "chunk": chunk_idx,
            "object": object_id,
            "action": action,
            "new_state": new_state,
        })

    def get_state_description(self, chunk_idx: int) -> str:
        """Generate a text description of all tracked object states.

        This text is prepended to the event description and fed into T5,
        giving the model awareness of current object states.
        """
        if not self.objects:
            return ""

        parts = []
        for obj in self.objects.values():
            age = chunk_idx - obj.last_interact_time
            parts.append(
                f"{obj.object_type} is {obj.state} "
                f"at {obj.location} (changed {age} steps ago)"
            )
        return "Object states: " + "; ".join(parts) + "."

    def get_changed_objects_near(self, camera_pos, radius: float = 5.0) -> List[ObjectState]:
        """Retrieve objects whose states have changed within a spatial radius."""
        # Used during inference to condition generation on nearby changed objects
        return [obj for obj in self.objects.values() if obj.bbox is not None]

    def serialize(self) -> str:
        return json.dumps({
            oid: {
                "type": obj.object_type,
                "state": obj.state,
                "location": obj.location,
                "last_action": obj.last_action,
                "last_interact_time": obj.last_interact_time,
            } for oid, obj in self.objects.items()
        })
```

### 3.4 Integration into YUME1.5 Inference Loop

In the autoregressive generation loop (`wan23/image2video.py` or equivalent inference script):

```python
osg = ObjectStateGraph()

for chunk_idx in range(num_chunks):
    # 1) Build caption with object state context
    state_desc = osg.get_state_description(chunk_idx)
    full_caption = base_caption + movement_desc + camera_desc + interaction_desc + state_desc

    # 2) Encode caption (state_desc is dynamic, must encode each chunk)
    #    Movement/camera descriptions are pre-cached (unchanged from YUME1.5)
    text_embedding = t5_encode(full_caption)

    # 3) Generate chunk
    chunk_frames = dit_generate(text_embedding, history_tokens, noise)

    # 4) If interaction action was issued, update OSG
    if current_interaction != "none":
        osg.update(
            object_id=target_obj_id,
            object_type=target_obj_type,
            action=current_interaction,
            new_state=predict_new_state(current_interaction, current_state),
            new_location=predict_new_location(current_interaction),
            chunk_idx=chunk_idx,
        )
```

### 3.5 Training Data Format for Object State

Each training sample metadata JSON:

```json
{
    "video_path": "ep_001234.mp4",
    "num_frames": 128,
    "fps": 16,
    "chunks": [
        {
            "chunk_idx": 0,
            "start_frame": 0,
            "end_frame": 32,
            "movement": "W",
            "camera": "→",
            "interaction": "none",
            "object_states": {}
        },
        {
            "chunk_idx": 1,
            "start_frame": 32,
            "end_frame": 64,
            "movement": "None",
            "camera": "·",
            "interaction": "pickup",
            "target_object": "Apple_01",
            "target_object_type": "Apple",
            "object_states": {
                "Apple_01": {"state": "in_hand", "location": "agent_hand"}
            }
        },
        {
            "chunk_idx": 2,
            "start_frame": 64,
            "end_frame": 96,
            "movement": "W",
            "camera": "←",
            "interaction": "none",
            "object_states": {
                "Apple_01": {"state": "in_hand", "location": "agent_hand"}
            }
        },
        {
            "chunk_idx": 3,
            "start_frame": 96,
            "end_frame": 128,
            "movement": "None",
            "camera": "·",
            "interaction": "place",
            "target_object": "Apple_01",
            "target_receptacle": "CounterTop_02",
            "object_states": {
                "Apple_01": {"state": "on_surface", "location": "CounterTop_02"}
            }
        }
    ]
}
```

---

## 4. Innovation Direction 3: Hierarchical Interaction Generation

### 4.1 Motivation

A single interaction (e.g., "chop a tree") spans multiple video chunks and involves a temporal sequence of visual changes:

```
Chunk 1: Approach tree, raise axe
Chunk 2: Axe strikes tree, wood chips fly
Chunk 3: Tree begins to tilt
Chunk 4: Tree falls, logs appear
```

Neither YUME1.5 nor HY-WorldPlay can plan such multi-chunk causal sequences. Their text events are single-shot descriptions applied to one chunk.

### 4.2 Design: LLM-Powered Interaction Planner

```
User Input: "hit tree"
        │
        ▼
┌───────────────────────────┐
│   Interaction Planner     │   (LLM, e.g., GPT-4 / InternVL)
│   (Zero-shot, no train)   │
│                           │
│   Input:                  │
│     - action: "hit"       │
│     - target: "tree"      │
│     - current state:      │
│       "intact"            │
│                           │
│   Output:                 │
│     Chunk 1 event:        │
│       "Agent raises axe   │
│        toward the tree"   │
│     Chunk 2 event:        │
│       "Axe strikes tree,  │
│        wood chips scatter" │
│     Chunk 3 event:        │
│       "Tree begins to     │
│        crack and tilt"    │
│     Chunk 4 event:        │
│       "Tree falls to the  │
│        ground with logs"  │
│     State transition:     │
│       "intact" → "chopped"│
└───────────────────────────┘
        │
        ▼
   Each chunk event → YUME1.5 Event Description
   (feeds into T5 text encoder per chunk)
```

### 4.3 Implementation

#### New file: `fastvideo/planner/interaction_planner.py`

```python
import json
from typing import List, Dict, Optional

PLAN_PROMPT_TEMPLATE = """You are a video generation planner for an interactive world simulator.

Given an interaction action and target object, generate a sequence of 2-4 chunk descriptions
that describe the visual progression of the interaction. Each chunk is ~2 seconds of video.

Rules:
- Each description should be a single sentence describing what is visually happening
- Descriptions must follow causal temporal order
- The first chunk shows the action beginning
- The last chunk shows the result/aftermath
- Include specific visual details (particles, motion direction, deformation)

Action: {action}
Target: {target_object}
Current State: {current_state}
Scene Context: {scene_context}

Output as JSON array of objects with "chunk_event" and "object_state" keys:
"""

COMMON_INTERACTION_PLANS = {
    ("pickup", "Apple"): [
        {"chunk_event": "Agent's hand reaches toward the apple on the table",
         "object_state": "intact"},
        {"chunk_event": "Agent grasps and lifts the apple off the surface",
         "object_state": "in_hand"},
    ],
    ("open", "Door"): [
        {"chunk_event": "Agent's hand pushes on the door handle",
         "object_state": "closed"},
        {"chunk_event": "The door swings open revealing the room beyond",
         "object_state": "open"},
    ],
    ("hit", "Tree"): [
        {"chunk_event": "Agent swings the axe toward the tree trunk",
         "object_state": "intact"},
        {"chunk_event": "Axe strikes the tree, wood chips scatter from the impact",
         "object_state": "damaged"},
        {"chunk_event": "The tree trunk cracks and the tree begins to lean",
         "object_state": "falling"},
        {"chunk_event": "Tree crashes to the ground, logs and branches scatter",
         "object_state": "chopped"},
    ],
    ("pet", "Cat"): [
        {"chunk_event": "Agent reaches hand toward the cat",
         "object_state": "idle"},
        {"chunk_event": "Agent gently strokes the cat's back, the cat purrs and arches",
         "object_state": "being_petted"},
        {"chunk_event": "The cat relaxes and nuzzles against the agent's hand",
         "object_state": "happy"},
    ],
}


class InteractionPlanner:
    """Plans multi-chunk event sequences for interactions.

    Uses a combination of:
    1. Pre-defined templates for common interactions (fast, deterministic)
    2. LLM fallback for novel interactions (flexible, slower)
    """

    def __init__(self, use_llm: bool = False, llm_model=None):
        self.use_llm = use_llm
        self.llm_model = llm_model

    def plan(self, action: str, target_object: str,
             current_state: str = "intact",
             scene_context: str = "") -> List[Dict]:
        """Generate a multi-chunk interaction plan.

        Args:
            action: Interaction verb (pickup, hit, open, pet, ...)
            target_object: Object type name
            current_state: Current object state
            scene_context: Scene description for context

        Returns:
            List of dicts with 'chunk_event' and 'object_state' per chunk.
        """
        key = (action, target_object)
        if key in COMMON_INTERACTION_PLANS:
            return COMMON_INTERACTION_PLANS[key]

        # Try generic templates based on action type
        generic = self._get_generic_plan(action, target_object, current_state)
        if generic:
            return generic

        # LLM fallback
        if self.use_llm and self.llm_model is not None:
            return self._llm_plan(action, target_object, current_state, scene_context)

        # Default: single-chunk event
        return [{"chunk_event": f"Agent {action}s the {target_object}",
                 "object_state": current_state}]

    def _get_generic_plan(self, action, target, state) -> Optional[List[Dict]]:
        """Generic templates by action type."""
        templates = {
            "pickup": [
                {"chunk_event": f"Agent reaches toward the {target}",
                 "object_state": state},
                {"chunk_event": f"Agent grasps and lifts the {target}",
                 "object_state": "in_hand"},
            ],
            "place": [
                {"chunk_event": f"Agent lowers the {target} onto the surface",
                 "object_state": "in_hand"},
                {"chunk_event": f"Agent releases the {target} onto the surface",
                 "object_state": "on_surface"},
            ],
            "open": [
                {"chunk_event": f"Agent reaches for and pushes the {target}",
                 "object_state": "closed"},
                {"chunk_event": f"The {target} swings open",
                 "object_state": "open"},
            ],
            "close": [
                {"chunk_event": f"Agent pushes the {target} shut",
                 "object_state": "open"},
                {"chunk_event": f"The {target} closes with a click",
                 "object_state": "closed"},
            ],
            "hit": [
                {"chunk_event": f"Agent swings at the {target}",
                 "object_state": state},
                {"chunk_event": f"The impact strikes the {target}, debris scatters",
                 "object_state": "damaged"},
                {"chunk_event": f"The {target} breaks apart",
                 "object_state": "broken"},
            ],
            "toggle_on": [
                {"chunk_event": f"Agent presses the switch on the {target}",
                 "object_state": "off"},
                {"chunk_event": f"The {target} turns on and begins operating",
                 "object_state": "on"},
            ],
        }
        return templates.get(action)

    def _llm_plan(self, action, target, state, context) -> List[Dict]:
        """Use LLM for novel interaction planning."""
        prompt = PLAN_PROMPT_TEMPLATE.format(
            action=action, target_object=target,
            current_state=state, scene_context=context,
        )
        response = self.llm_model.generate(prompt, max_tokens=512)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [{"chunk_event": f"Agent {action}s the {target}",
                     "object_state": state}]
```

### 4.4 Integration into Autoregressive Inference

```python
planner = InteractionPlanner(use_llm=False)
osg = ObjectStateGraph()

chunk_idx = 0
while generating:
    action = get_user_action()  # keyboard/mouse + interaction

    if action.interaction != "none":
        # Plan multi-chunk sequence
        plan = planner.plan(
            action=action.interaction,
            target_object=action.target_object,
            current_state=osg.get_object_state(action.target_object),
        )
        # Execute plan across multiple chunks
        for step in plan:
            event_desc = step["chunk_event"]
            caption = base_caption + movement_desc + camera_desc + event_desc
            caption += osg.get_state_description(chunk_idx)

            chunk = generate_chunk(caption, history_tokens)
            osg.update(action.target_object, ..., step["object_state"], chunk_idx)
            chunk_idx += 1
    else:
        # Normal navigation (unchanged from YUME1.5)
        caption = base_caption + movement_desc + camera_desc
        caption += osg.get_state_description(chunk_idx)
        chunk = generate_chunk(caption, history_tokens)
        chunk_idx += 1
```

---

## 5. Innovation Direction 4: Interaction-Aware RL Post-Training

### 5.1 Motivation

HY-WorldPlay's WorldCompass uses two reward signals:
1. Action following score (does the generated video match the intended camera movement?)
2. Visual quality score (is the video aesthetically good?)

Neither reward captures **interaction quality**. We need interaction-specific rewards.

### 5.2 Reward Function Design

```python
class InteractionRewardModel:
    """Compute rewards for interaction quality in generated videos."""

    def __init__(self, vlm_model, object_detector):
        self.vlm = vlm_model           # e.g., InternVL3
        self.detector = object_detector  # e.g., GroundingDINO

    def compute_reward(self, frames_before, frames_after,
                       action, target_object, expected_state) -> Dict[str, float]:
        """
        Args:
            frames_before: Video frames from the chunk before interaction
            frames_after: Video frames from the chunk after interaction
            action: Interaction action (e.g., "pickup")
            target_object: Target object type (e.g., "Apple")
            expected_state: Expected state after interaction (e.g., "in_hand")

        Returns:
            Dict of reward components, each in [0, 1]
        """
        rewards = {}

        # R1: Interaction Causality — did the interaction visibly happen?
        prompt = (
            f"In this video, the agent performed '{action}' on '{target_object}'. "
            f"Rate from 0-10 how clearly the interaction is visible and correct."
        )
        causality_score = self.vlm.score(frames_after, prompt) / 10.0
        rewards["causality"] = causality_score

        # R2: State Consistency — does the post-interaction state match expectation?
        prompt = (
            f"After the agent {action}ed the {target_object}, "
            f"is the {target_object} now {expected_state}? Rate 0-10."
        )
        state_score = self.vlm.score(frames_after, prompt) / 10.0
        rewards["state_consistency"] = state_score

        # R3: Object Permanence — on revisit, is the changed state preserved?
        # (Computed across longer rollouts, not per-chunk)

        # R4: Physical Plausibility — does the motion look physically correct?
        prompt = (
            f"Does the object motion in this interaction look physically realistic? "
            f"Rate 0-10 for physical plausibility."
        )
        physics_score = self.vlm.score(frames_after, prompt) / 10.0
        rewards["physics"] = physics_score

        # R5: Visual Quality (same as WorldCompass)
        rewards["visual_quality"] = self._vbench_score(frames_after)

        # Composite reward
        rewards["total"] = (
            0.3 * rewards["causality"]
            + 0.3 * rewards["state_consistency"]
            + 0.2 * rewards["physics"]
            + 0.2 * rewards["visual_quality"]
        )
        return rewards

    def _vbench_score(self, frames):
        """Visual quality score from VBench metrics."""
        # Use VBench aesthetic + imaging quality
        pass
```

### 5.3 RL Training Pipeline

Adapting WorldCompass's Clip-Level Rollout for interactions:

```python
def interaction_rl_training_step(model, planner, reward_model, osg):
    """One RL training step with interaction rollout."""
    # 1) Sample an interaction scenario
    action, target, scene_image = sample_interaction_scenario()

    # 2) Plan the interaction sequence
    plan = planner.plan(action, target)

    # 3) Rollout: generate chunks following the plan
    generated_chunks = []
    for step in plan:
        caption = build_caption(step["chunk_event"], osg)
        chunk = model.generate_chunk(caption, history=generated_chunks)
        generated_chunks.append(chunk)

    # 4) Compute interaction-aware rewards
    rewards = reward_model.compute_reward(
        frames_before=generated_chunks[0],
        frames_after=generated_chunks[-1],
        action=action,
        target_object=target,
        expected_state=plan[-1]["object_state"],
    )

    # 5) Policy gradient update (similar to WorldCompass's DiffusionNFT)
    loss = -rewards["total"] * log_prob(generated_chunks)
    loss.backward()
```

### 5.4 Training Data for RL

RL training does **not** require paired ground truth — it uses the reward model's scores. However, we need:

1. A set of **scene images** to start from (can reuse UE5/AI2-THOR scenes)
2. A set of **interaction commands** to sample from
3. The reward model (VLM + object detector)

---

## 6. Innovation Direction 5: Affordance-Conditioned Generation

### 6.1 Motivation

Not all objects can be interacted with, and not all interactions are valid. We need to model **visual affordance**: what can be done with each object in view.

### 6.2 Design: Affordance Map as Spatial Conditioning

```
Input frame → Object Detector (GroundingDINO / SAM)
    → For each detected object:
        VLM predicts affordances:
            "Apple" → [pickable, throwable, sliceable]
            "Door" → [openable, closable]
            "Cat" → [pettable]
            "Wall" → [none]
    → Affordance Map (H × W spatial map with affordance labels)
    → Encode as additional cross-attention tokens in DiT
```

### 6.3 Implementation: Affordance Encoder

```python
import torch
import torch.nn as nn


class AffordanceEncoder(nn.Module):
    """Encodes affordance maps into tokens for DiT cross-attention.

    The affordance map is a spatial tensor (H, W) where each pixel
    stores an affordance class ID. This is converted to a set of
    affordance tokens that are concatenated with text embeddings
    in the cross-attention layer.
    """

    def __init__(self, num_affordance_classes: int = 16, embed_dim: int = 4096,
                 num_tokens: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_affordance_classes, embed_dim)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((num_tokens, 1)),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, affordance_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            affordance_map: (B, H, W) integer tensor of affordance class IDs

        Returns:
            (B, num_tokens, embed_dim) affordance tokens for cross-attention
        """
        x = self.embedding(affordance_map)     # (B, H, W, D)
        x = x.permute(0, 3, 1, 2)              # (B, D, H, W)
        x = self.spatial_proj(x)                # (B, D, num_tokens, 1)
        x = x.squeeze(-1).permute(0, 2, 1)     # (B, num_tokens, D)
        return self.proj(x)
```

#### Integration in DiT Block (modify `wan23/modules/model.py`, WanAttentionBlock.forward):

```python
# In WanAttentionBlock.forward, after cross-attention with text:
# Original:
#   x = x + cross_attn(norm(x), context, context_lens)
# New:
#   affordance_tokens = affordance_encoder(affordance_map)
#   context_with_affordance = torch.cat([context, affordance_tokens], dim=1)
#   context_lens_with_affordance = context_lens + num_affordance_tokens
#   x = x + cross_attn(norm(x), context_with_affordance, context_lens_with_affordance)
```

### 6.4 Affordance Class Definitions

```python
AFFORDANCE_CLASSES = {
    0: "none",            # Background, non-interactable
    1: "pickable",        # Can be picked up (apple, cup, book)
    2: "openable",        # Can be opened (door, drawer, fridge)
    3: "toggleable",      # Can be toggled (lamp, TV, stove)
    4: "sliceable",       # Can be sliced (bread, apple, tomato)
    5: "breakable",       # Can be broken (vase, window, egg)
    6: "pushable",        # Can be pushed (box, cart, ball)
    7: "pettable",        # Can be petted (cat, dog)
    8: "sittable",        # Can be sat on (chair, couch, bench)
    9: "hittable",        # Can be hit/chopped (tree, rock)
    10: "fillable",       # Can be filled (cup, bowl, pot)
    11: "climbable",      # Can be climbed (ladder, stairs)
    12: "throwable",      # Can be thrown (ball, stone)
    13: "wearable",       # Can be worn (hat, glasses)
    14: "rideable",       # Can be ridden (horse, bike)
    15: "readable",       # Can be read (book, sign, screen)
}
```

---

## 7. Data Engine: Interaction Data Pipeline

### 7.1 Existing Infrastructure

You already have a working data collection pipeline in `interactive_world_gen/`:

```
interactive_world_gen/
├── src/
│   ├── actions/
│   │   ├── action_space.py     # Full interaction action space (already defined!)
│   │   └── scenarios.py        # Interaction scenarios
│   ├── simulator/
│   │   ├── thor_engine.py      # AI2-THOR engine
│   │   └── scene_manager.py    # Scene management
│   ├── recorder/
│   │   └── video_recorder.py   # Video + metadata recording
│   ├── pipeline/
│   │   └── data_collector.py   # Main collection pipeline
│   └── prompt/
│       └── prompt_generator.py # NL prompt generation
└── ue5_gen/                     # UE5 data generation
```

Your `action_space.py` already defines all the interaction actions we need:

```python
# Already defined in interactive_world_gen/src/actions/action_space.py
INTERACTION_ACTIONS = [
    ActionType.PICKUP,      # "picks up {object}"
    ActionType.PUT,         # "places {object} on {target}"
    ActionType.DROP,        # "drops {object}"
    ActionType.THROW,       # "throws {object}"
    ActionType.PUSH,        # "pushes {object}"
    ActionType.OPEN,        # "opens {object}"
    ActionType.CLOSE,       # "closes {object}"
    ActionType.TOGGLE_ON,   # "turns on {object}"
    ActionType.TOGGLE_OFF,  # "turns off {object}"
    ActionType.SLICE,       # "slices {object}"
]
```

### 7.2 Target Data Format (YUME1.5 Compatible)

```
interaction_data/
├── pickup_Apple/
│   ├── ep_000001.mp4                 # First-person RGB video, 16fps, 704×1280
│   ├── ep_000001.txt                 # Per-chunk action annotations (YUME format)
│   ├── ep_000001.npy                 # Camera c2w poses [N, 4, 4]
│   ├── ep_000001_meta.json           # Full interaction metadata
│   ├── ep_000001_object_states.json  # Object state timeline
│   └── ep_000001_affordance.npz      # Affordance maps per frame (optional)
├── open_Door/
│   ├── ...
├── hit_Tree/
│   ├── ...
├── pet_Cat/
│   ├── ...
└── manifest.csv                       # Master index of all episodes
```

### 7.3 Individual File Formats

#### 7.3.1 Video File (`ep_XXXXXX.mp4`)

```yaml
Format: MP4 (H.264)
Resolution: 704 × 1280 (matching YUME1.5 training)
FPS: 16 (matching YUME1.5)
Duration: 4-16 seconds (64-256 frames, 2-8 chunks of 32 frames)
Perspective: First-person (egocentric)
Content: Full interaction sequence (approach → interact → aftermath)
```

#### 7.3.2 Action Annotation (`ep_XXXXXX.txt`)

YUME1.5-compatible format with interaction extensions:

```
Keys: W
Mouse: →
Interaction: pickup
Target Object: Apple
Target Receptacle:
Object State Before: on_table
Object State After: in_hand
Start Frame: 0
End Frame: 128
Chunk Events:
  0-32: Agent walks toward the apple on the kitchen counter
  32-64: Agent reaches out hand toward the apple
  64-96: Agent grasps and lifts the apple from the counter
  96-128: Agent holds the apple, examining it
```

#### 7.3.3 Camera Poses (`ep_XXXXXX.npy`)

Unchanged from YUME1.5:

```python
# Shape: [N_frames, 4, 4], float32
# Content: Camera-to-world (c2w) transformation matrices
# Coordinate system: OpenGL (right-handed, Y-up)
# Aligned to first frame (T[0] = I)
poses = np.load("ep_000001.npy")
assert poses.shape == (128, 4, 4)
assert np.allclose(poses[0], np.eye(4))
```

#### 7.3.4 Full Metadata (`ep_XXXXXX_meta.json`)

```json
{
    "episode_id": "ep_000001",
    "scene": "FloorPlan1",
    "scene_type": "kitchen",
    "scenario": "pickup_apple",
    "total_frames": 128,
    "fps": 16,
    "resolution": [704, 1280],
    "interaction_summary": {
        "action": "pickup",
        "target": "Apple_01",
        "target_type": "Apple",
        "success": true,
        "start_frame": 48,
        "end_frame": 96
    },
    "action_sequence": [
        {
            "frame": 0,
            "action": "MoveAhead",
            "movement": "W",
            "camera": "·",
            "interaction": "none",
            "agent_position": [1.25, 0.9, -0.5],
            "agent_rotation": [0, 180, 0]
        },
        {
            "frame": 32,
            "action": "RotateRight",
            "movement": "None",
            "camera": "→",
            "interaction": "none",
            "agent_position": [1.25, 0.9, 0.0],
            "agent_rotation": [0, 210, 0]
        },
        {
            "frame": 64,
            "action": "PickupObject",
            "movement": "None",
            "camera": "·",
            "interaction": "pickup",
            "target_object_id": "Apple_01",
            "target_object_type": "Apple",
            "agent_position": [1.25, 0.9, 0.5],
            "agent_rotation": [0, 210, 0]
        },
        {
            "frame": 96,
            "action": "MoveAhead",
            "movement": "W",
            "camera": "·",
            "interaction": "none",
            "held_object": "Apple_01",
            "agent_position": [1.25, 0.9, 1.0],
            "agent_rotation": [0, 210, 0]
        }
    ],
    "object_states_timeline": [
        {
            "frame": 0,
            "objects": {
                "Apple_01": {"state": "on_surface", "location": "CounterTop_01", "visible": true},
                "Fridge_01": {"state": "closed", "location": "fixed", "visible": true}
            }
        },
        {
            "frame": 64,
            "objects": {
                "Apple_01": {"state": "in_hand", "location": "agent_hand", "visible": true},
                "Fridge_01": {"state": "closed", "location": "fixed", "visible": true}
            }
        }
    ],
    "caption": "This video depicts a first-person interactive scene in a modern kitchen. Person moves forward (W). Camera remains still (·). Agent picks up the apple from the counter.",
    "event_captions": [
        "Agent walks toward the kitchen counter with an apple",
        "Agent reaches toward the red apple on the counter",
        "Agent picks up the apple, lifting it from the surface",
        "Agent holds the apple while moving forward"
    ]
}
```

#### 7.3.5 Object State Timeline (`ep_XXXXXX_object_states.json`)

```json
{
    "tracked_objects": ["Apple_01", "Fridge_01", "Knife_01"],
    "states": [
        {
            "frame": 0,
            "Apple_01": {"state": "on_surface", "position": [1.5, 0.85, 0.3],
                         "bbox_2d": [420, 280, 510, 350], "visible": true},
            "Fridge_01": {"state": "closed", "position": [2.0, 0.0, -0.5],
                          "bbox_2d": [100, 50, 350, 600], "visible": true},
            "Knife_01": {"state": "on_surface", "position": [1.3, 0.85, 0.1],
                         "bbox_2d": null, "visible": false}
        },
        {
            "frame": 64,
            "Apple_01": {"state": "in_hand", "position": null,
                         "bbox_2d": [600, 400, 680, 460], "visible": true},
            "Fridge_01": {"state": "closed", "position": [2.0, 0.0, -0.5],
                          "bbox_2d": [50, 50, 300, 600], "visible": true},
            "Knife_01": {"state": "on_surface", "position": [1.3, 0.85, 0.1],
                         "bbox_2d": [380, 290, 430, 310], "visible": true}
        }
    ]
}
```

#### 7.3.6 Master Manifest (`manifest.csv`)

```csv
episode_id,video_path,txt_path,npy_path,meta_path,scene,interaction,target_object,num_frames,success
ep_000001,pickup_Apple/ep_000001.mp4,pickup_Apple/ep_000001.txt,pickup_Apple/ep_000001.npy,pickup_Apple/ep_000001_meta.json,FloorPlan1,pickup,Apple,128,true
ep_000002,open_Door/ep_000002.mp4,open_Door/ep_000002.txt,open_Door/ep_000002.npy,open_Door/ep_000002_meta.json,FloorPlan3,open,Door,96,true
ep_000003,hit_Tree/ep_000003.mp4,hit_Tree/ep_000003.txt,hit_Tree/ep_000003.npy,hit_Tree/ep_000003_meta.json,OutdoorScene1,hit,Tree,256,true
```

### 7.4 Data Collection from AI2-THOR (Existing Pipeline)

Modify `interactive_world_gen/src/pipeline/data_collector.py` to output in YUME format:

```python
def collect_episode_yume_format(self, episode_idx: int) -> Dict:
    """Collect one episode and save in YUME1.5-compatible format."""
    episode_id = f"ep_{episode_idx:06d}"

    # 1) Sample scene and scenario (existing code)
    scene = self.scene_manager.sample_scene()
    self.engine.reset(scene)
    scenario = self.scenario_manager.sample_scenario(scene)

    # 2) Execute actions and record
    self.recorder.start_episode(episode_id)
    frames, actions, agent_states, object_states = [], [], [], []
    c2w_matrices = []

    for step_action in scenario.action_sequence:
        # Execute in simulator
        event = self.engine.step(step_action.to_thor_dict())
        frame = event.frame     # RGB numpy array
        depth = event.depth_frame

        # Record frame
        self.recorder.add_frame(frame)
        frames.append(frame)

        # Record action in YUME format
        movement = self._action_to_yume_movement(step_action)
        camera = self._action_to_yume_camera(step_action)
        interaction = self._action_to_yume_interaction(step_action)

        actions.append({
            "movement": movement,
            "camera": camera,
            "interaction": interaction,
            "target_object": step_action.target_object,
        })

        # Record camera pose
        agent = event.metadata["agent"]
        c2w = self._agent_to_c2w(agent)  # Convert THOR agent state to 4x4 c2w
        c2w_matrices.append(c2w)

        # Record object states
        visible_objects = self.engine.get_visible_objects()
        object_states.append(self._snapshot_object_states(visible_objects))

    self.recorder.end_episode()

    # 3) Save in YUME format
    c2w_array = np.stack(c2w_matrices)  # [N, 4, 4]
    c2w_array = normalize_c2w_matrices(c2w_array)

    # Determine dominant action for this episode
    dominant_movement = self._get_dominant(actions, "movement")
    dominant_camera = self._get_dominant(actions, "camera")
    dominant_interaction = self._get_dominant(actions, "interaction")

    # Write TXT
    txt_content = (
        f"Keys: {dominant_movement}\n"
        f"Mouse: {dominant_camera}\n"
        f"Interaction: {dominant_interaction}\n"
        f"Target Object: {actions[-1].get('target_object', '')}\n"
        f"Start Frame: 0\n"
        f"End Frame: {len(frames)}\n"
    )

    # Save all files
    category = f"{dominant_interaction}_{actions[-1].get('target_object', 'none')}"
    output_dir = self.output_base / category
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{episode_id}.npy", c2w_array)
    (output_dir / f"{episode_id}.txt").write_text(txt_content)
    # Video already saved by recorder

    return {"episode_id": episode_id, "category": category}
```

### 7.5 Data Collection from UE5 (Your Existing Pipeline)

Your UE5 pipeline (`ue5_gen/`) already generates videos. Extend it to:

1. Record object interaction events (spawn/destroy/state change triggers in UE5 Blueprints)
2. Export per-frame object bounding boxes (UE5 SceneCapture + custom render pass)
3. Export camera c2w matrices (from UE5 camera actor transform)

### 7.6 Data Scale Targets

| Stage | Source | Interaction Types | Clips | Purpose |
|-------|--------|-------------------|-------|---------|
| Stage 1 | AI2-THOR | pickup, place, open, close, toggle | 5,000 | Core interaction learning |
| Stage 2 | UE5 Synthetic | hit, push, throw, slice, break | 5,000 | Physics-heavy interactions |
| Stage 3 | UE5 Outdoor | pet, ride, climb, chop tree | 3,000 | Outdoor/animal interactions |
| Stage 4 | VLM-annotated web video | Various | 10,000 | Generalization |
| **Total** | | **15+ types** | **~23,000** | |

Note: YUME1.5's Event Dataset achieved text-event generation with only 4,000 clips. Our 23K target is conservative and should be sufficient.

---

## 8. Training Recipe & Roadmap

### 8.1 Training Stages

```
Stage 0: YUME1.5 Foundation Model (provided, Wan2.2-5B, 10K iters)
    │
    ▼
Stage 1: Interaction Fine-tuning (5K-10K iters)
    - Data: 23K interaction clips + original Sekai data (prevent forgetting)
    - Alternating: 50% interaction data, 30% Sekai navigation, 20% synthetic T2V
    - Loss: Flow Matching (unchanged)
    - New: Extended action vocabulary in captions
    │
    ▼
Stage 2: Object State Memory Training (3K-5K iters)
    - Data: Same as Stage 1, but with object state descriptions in captions
    - New: ObjectStateGraph text descriptions prepended to captions
    - No architecture change — pure text conditioning
    │
    ▼
Stage 3 (Optional): Affordance Conditioning (3K iters)
    - Data: Same + affordance map annotations
    - New: AffordanceEncoder module (small, ~50M params)
    - Architecture change: additional cross-attention tokens
    │
    ▼
Stage 4: Self-Forcing + TSCM Distillation (600 iters, same as YUME1.5)
    - Distill multi-step model into 4-step generator
    - Include interaction data in distillation
    │
    ▼
Stage 5 (Optional): Interaction RL Post-Training (200-500 iters)
    - Clip-level rollout with interaction scenarios
    - VLM-based interaction rewards
```

### 8.2 Training Script Modification

Modify `scripts/finetune/finetune-5b.sh`:

```bash
#!/usr/bin/bash
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node 8 --master_port 9607 \
    train-5b.py \
    --seed 42 \
    --train_batch_size=1 \
    --dataloader_num_workers 32 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=10000 \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=200 \
    --checkpoints_total_limit=5 \
    --validation_steps 100 \
    --allow_tf32 \
    --t5_cpu \
    --output_dir="./outputs_interactive_world" \
    --fps=16 \
    --interaction_data_dir="./interaction_data" \
    --sekai_data_dir="./sekai_data" \
    --synthetic_data_dir="./synthetic_data" \
    --interaction_data_ratio=0.5 \
    --sekai_data_ratio=0.3 \
    --synthetic_data_ratio=0.2 \
    --enable_object_state_caption \
    --enable_interaction_actions
```

### 8.3 Dataset Class Modification

New dataset class that handles interaction data:

```python
class InteractiveWorldDataset(StableVideoAnimationDataset):
    """Extended dataset supporting interaction actions and object states.

    Inherits from YUME1.5's StableVideoAnimationDataset and adds:
    - Interaction action parsing
    - Object state description injection
    - Multi-chunk event caption support
    """

    def __init__(self, *args, interaction_data_dir=None,
                 enable_object_state=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_object_state = enable_object_state

        if interaction_data_dir:
            self._load_interaction_data(interaction_data_dir)

    def _load_interaction_data(self, data_dir):
        """Load interaction episodes into vid_meta."""
        for subdir in glob.glob(os.path.join(data_dir, '*/')):
            mp4_files = glob.glob(os.path.join(subdir, '*.mp4'))
            for mp4_path in mp4_files:
                base = os.path.splitext(os.path.basename(mp4_path))[0]
                txt_path = os.path.join(subdir, f"{base}.txt")
                npy_path = os.path.join(subdir, f"{base}.npy")
                meta_path = os.path.join(subdir, f"{base}_meta.json")

                if os.path.exists(txt_path):
                    keys, mouse, interaction, target_obj, target_recep, \
                        state_before, state_after = parse_txt_file_extended(txt_path)

                    # Load event captions if available
                    event_captions = []
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                            event_captions = meta.get("event_captions", [])

                    self.vid_meta.append((
                        mp4_path, base, [keys], [mouse],
                        npy_path, 0, -1, False, None,
                        # New fields:
                        [interaction], [target_obj], [target_recep],
                        [state_before], [state_after], event_captions,
                    ))

    def get_sample(self, index):
        meta = self.vid_meta[index]

        # Unpack (handle both old and new format)
        if len(meta) > 9:
            video_path, videoid, keys, mouse, npz_path, \
                start_frame, end_frame, flag, full_mp4_1, \
                interaction, target_obj, target_recep, \
                state_before, state_after, event_captions = meta
        else:
            # Old YUME format — no interaction
            video_path, videoid, keys, mouse, npz_path, \
                start_frame, end_frame, flag, full_mp4_1 = meta
            interaction = ["none"]
            target_obj = [""]
            state_before = [""]
            state_after = [""]
            event_captions = []

        # Build caption
        caption = "This video depicts a first-person interactive scene."
        caption += vocab_movement[keys[0]]
        caption += vocab_camera[mouse[0]]

        # Add interaction description
        if interaction[0] != "none":
            interact_desc = vocab_interaction[interaction[0]].format(
                object=target_obj[0] or "the object",
                target=target_recep[0] if target_recep else "the surface",
                animal=target_obj[0] or "the animal",
            )
            caption += interact_desc

        # Add object state context
        if self.enable_object_state and state_before[0]:
            caption += f"Before: {target_obj[0]} is {state_before[0]}. "
            caption += f"After: {target_obj[0]} becomes {state_after[0]}."

        # Randomly use chunk-level event caption (for hierarchical training)
        if event_captions and random.random() < 0.5:
            chunk_event = random.choice(event_captions)
            caption += f" Event: {chunk_event}"

        # ... rest is same as original get_sample() ...
        # Load video, apply transforms, etc.
```

### 8.4 Evaluation Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Interaction Success Rate** | Does the visual output match the intended interaction? | VLM scoring on 500 test scenarios |
| **State Consistency** | After interaction, does the object state persist? | Compare frames at t+1, t+5, t+10 chunks |
| **Object Permanence** | On revisit (cycle trajectory), is changed state preserved? | Same as HY-WorldPlay's long-term eval |
| **Interaction Following (IF)** | Extension of YUME-Bench's IF metric to interactions | Automated VLM assessment |
| **Physical Plausibility** | Does interaction motion look physically correct? | Human evaluation (A/B test) |
| **Visual Quality** | VBench metrics (unchanged) | SC, BC, MS, AQ, IQ from VBench |
| **Navigation Quality** | Ensure no regression on navigation | Original YUME-Bench metrics |

---

## Summary: Paper Story & Core Contributions

### Title (Working)

**"InteractiveWorld: From Navigation to Object Interaction in Video World Models"**

### Core Story

> Current video world models (YUME, WorldPlay, Matrix-Game) can generate visually stunning interactive worlds, but they only support *observing* the world through navigation. We present InteractiveWorld, the first video world model that enables *changing* the world through object-level interactions — picking up objects, opening doors, chopping trees, petting animals — while maintaining causal consistency of object states across long-horizon generation.

### Contribution List

1. **Extended Interaction Action Space**: First unified action vocabulary covering both navigation (WASD + camera) and object interactions (15+ interaction verbs), integrated as text conditioning with zero additional inference cost.

2. **Object State Memory (OSG)**: A novel object-centric memory system that tracks physical states of interacted objects and injects state awareness into the generation model, ensuring causal consistency (e.g., a chopped tree stays chopped on revisit).

3. **Hierarchical Interaction Planner**: An LLM-powered planner that decomposes complex interactions into multi-chunk causal event sequences, enabling temporally coherent interaction generation.

4. **Interaction Data Engine**: A large-scale synthetic interaction data pipeline based on AI2-THOR and UE5, producing 23K+ annotated interaction video clips with structured object state annotations.

5. **(Optional) Interaction-Aware RL**: VLM-based reward functions for interaction causality, state consistency, and physical plausibility, extending HY-WorldPlay's WorldCompass framework.
