from __future__ import annotations

import json
from typing import Any


QUESTION_TYPES = [
    "causal_motivational",
    "inferred_counting",
    "lateral_spatial",
    "motion_trajectory",
    "physical_environment",
    "relative_depth",
    "social_interaction",
    "vertical_spatial",
    "viewpoint_visibility",
    "other",
]


TYPE_GUIDANCE = {
    "causal_motivational": (
        "Causal and Motivational Reasoning: build a before -> trigger -> reaction/state-change chain. "
        "Prefer visible temporal causes, contact, gaze, object state changes, and reactions over story priors."
    ),
    "inferred_counting": (
        "Inferred Counting: use a high-recall ledger of distinct entities/events across the whole clip. "
        "Include edge, partial, distant, occluded, and briefly visible candidates, then merge duplicates across time only when identity continuity is supported."
    ),
    "lateral_spatial": (
        "Lateral Spatial Reasoning: first choose image/screen, scene-world, or character-viewpoint coordinates. "
        "Unless wording explicitly says camera/viewer/screen/perspective or a character's view, use the visible scene relation."
    ),
    "motion_trajectory": (
        "Motion and Trajectory Dynamics: compare start, middle, and end positions. For repeated actions, count only complete cycles/events. "
        "For order wording, distinguish physical/scene order from order of appearance; use temporal order only when the question says appearance/shown/appears."
    ),
    "physical_environment": (
        "Physical and Environmental Context: inventory objects, surfaces, containers, terrain, lighting, weather, and affordances. "
        "Do not overfit to a close-up when wide context explains the answer."
    ),
    "relative_depth": (
        "Relative Depth and Proximity: use occlusion, floor/contact points, support surfaces, scale only when comparable, and latest co-visible span."
    ),
    "social_interaction": (
        "Social Interaction and Relationships: track people/characters, gaze, body orientation, turn-taking, approach/avoidance, touch, and shared attention."
    ),
    "vertical_spatial": (
        "Vertical Spatial Reasoning: distinguish image height, physical support/stacking, wall attachment, suspension, and depth occlusion."
    ),
    "viewpoint_visibility": (
        "Viewpoint and Visibility: separate what is visible to the camera from what a named character can see; check line of sight, occluders, and facing."
    ),
    "other": "Use the question wording to choose the closest applicable reasoning routine.",
}


def guidance_for_type(question_type: str | None) -> str:
    key = str(question_type or "other").strip().lower()
    return TYPE_GUIDANCE.get(key, TYPE_GUIDANCE["other"])


def question_block(question: dict[str, Any]) -> str:
    return json.dumps(
        {
            "question_id": question.get("question_id"),
            "question_text": question.get("question_text"),
            "options": question.get("options"),
            "question_start_time": question.get("question_start_time"),
            "question_stop_time": question.get("question_stop_time"),
        },
        ensure_ascii=False,
        indent=2,
    )


def scout_prompt(question: dict[str, Any], heuristic_type: str) -> str:
    return f"""You are a video-QA evidence planner. The attached image is a coarse timeline contact sheet.

Question:
{question_block(question)}

Heuristic question type guess: {heuristic_type}
Allowed type labels: {", ".join(QUESTION_TYPES)}

Nine reasoning routines:
{json.dumps(TYPE_GUIDANCE, ensure_ascii=False, indent=2)}

Think privately. Do not answer from a single impression unless the evidence is unmistakable.
Return exactly one JSON object:
{{
  "question_type": "one allowed type label",
  "answer_policy": "short rule for this question, including coordinate frame or count/event rules",
  "scout_observations": ["brief visible facts from the timeline"],
  "evidence_requests": [
    {{
      "kind": "dense_sheet|time_window|frames|crop",
      "label": "short label",
      "reason": "why this evidence is needed",
      "start_sec": 0.0,
      "end_sec": 1.0,
      "fps": 1.5,
      "times_sec": [0.0],
      "time_sec": 0.0,
      "box_pct": [0, 0, 100, 100]
    }}
  ],
  "tentative_answer_choice": null,
  "risk_notes": ["likely failure modes to audit"]
}}

Rules for evidence_requests:
- Use at most 4 requests.
- Use crop boxes as percentages [x1,y1,x2,y2], where 0,0 is top-left and 100,100 is bottom-right.
- For counting, request high-recall evidence for edge/partial/occluded candidates before deciding.
- For spatial questions, specify whether to use image layout, scene-world relation, character viewpoint, depth, or endpoint state.
- For motion/event counts, distinguish complete cycles from partial poses.
- Do not mention any hidden answer key or dataset statistics."""


def answer_prompt(question: dict[str, Any], scout: dict[str, Any], evidence_notes: list[dict[str, str]]) -> str:
    qtype = str(scout.get("question_type") or "other")
    return f"""You are solving one multiple-choice video question using an agentic evidence pass.

Question:
{question_block(question)}

Scout plan JSON:
{json.dumps(scout, ensure_ascii=False, indent=2)}

Attached evidence images:
{json.dumps(evidence_notes, ensure_ascii=False, indent=2)}

Question-type routine:
{guidance_for_type(qtype)}

Use private reasoning, then return only JSON. The public fields should be concise and auditable, not a long chain of thought.

General rules:
- The frames/contact sheets are chronological.
- Use only the clip evidence. Do not use outside knowledge.
- If the question explicitly says camera/viewer/screen/frame/perspective, answer from that coordinate frame. Otherwise use the visible scene relation requested by the wording.
- For counting, make a high-recall ledger first: include edge, partial, occluded, and briefly visible candidates; then merge duplicates across time. Do not answer from the maximum count in one frame unless the question asks for a simultaneous count.
- For motion, count only completed events/cycles when the noun implies a completed action.
- For endpoint wording, answer from the requested endpoint or latest co-visible/localizable endpoint span.
- For "from first to last" or "order" wording, do not assume shot order. If the question asks order based on entities in a scene and does not say "appearance", "shown", or "enters", use the physical scene/path order in the clearest supporting span.

Type-specific hard requirements:
- Counting objects: the evidence_ledger must enumerate candidates as C1, C2, ... with first/last evidence and a merge/exclude reason. If the options include a threshold such as "12 or more" or "6 or less", compute a count range and choose the option whose numeric range contains it. When in doubt between a conservative clear count and a higher count supported by partial/edge candidates, prefer the high-recall count if the candidates are visually identifiable.
- Counting events/actions: the evidence_ledger must enumerate completed cycles/events as E1, E2, ... with start and end evidence. Do not count a posture change, bounce, occluded continuation, or same continuous motion arc as a new event unless a full cycle completes.
- Rotation-cycle questions: a counted cycle needs a visibly supported full body/object rotation in a continuous visual span. Do not bridge a cycle across a cut, heavy occlusion, or offscreen interval. If the evidence supports either a lower count of clearly completed cycles or a higher count that depends on inferred hidden rotation, choose the lower clear count.
- Spatial/order: state the coordinate frame or ordering axis before choosing. For physical path/scene order, order entities along the visible path or layout, not by which close-up appeared first.

Return exactly one JSON object:
{{
  "question_id": "{question.get('question_id')}",
  "answer_choice": "one listed option letter",
  "confidence": 0.0,
  "evidence_ledger": [
    "compact itemized evidence; for counts use candidate/merge notes, for spatial use target-reference-frame notes"
  ],
  "rejected_options": [
    "brief note about the most tempting rejected option"
  ],
  "needs_verification": true,
  "verification_requests": [
    {{
      "kind": "dense_sheet|time_window|frames|crop",
      "label": "short label",
      "reason": "what uncertainty remains",
      "start_sec": 0.0,
      "end_sec": 1.0,
      "fps": 1.5,
      "times_sec": [0.0],
      "time_sec": 0.0,
      "box_pct": [0, 0, 100, 100]
    }}
  ]
}}"""


def verify_prompt(
    question: dict[str, Any],
    scout: dict[str, Any],
    answer: dict[str, Any],
    evidence_notes: list[dict[str, str]],
) -> str:
    qtype = str(scout.get("question_type") or "other")
    return f"""You are the final verifier for a multiple-choice video answer. Try to falsify the draft before accepting it.

Question:
{question_block(question)}

Scout plan:
{json.dumps(scout, ensure_ascii=False, indent=2)}

Draft answer:
{json.dumps(answer, ensure_ascii=False, indent=2)}

Attached evidence images:
{json.dumps(evidence_notes, ensure_ascii=False, indent=2)}

Question-type routine:
{guidance_for_type(qtype)}

Audit checklist:
- Causal/motivational: visible trigger precedes effect; do not invent motives without visual support.
- Counting objects: look for under-counting first, then duplicate over-counting. Edge/partial/distant objects count if visibly identifiable. Do not merge separate candidates just because they look similar; merge only with position/track continuity.
- Lateral spatial: re-check coordinate frame. Camera/viewer wording means image frame; otherwise use the scene relation implied by the question.
- Motion/trajectory: distinguish complete cycles from transient poses, and use before/after order. For rotations, require evidence of a full rotation cycle; if a continuous motion contains several poses but not several completed rotations, count the lower complete-cycle total.
- Rotation-cycle audit: reject any cycle whose start/end states are connected only by a cut, heavy occlusion, blur, or offscreen continuation. Count clear full rotations, not generic bounces or pose changes.
- Order/sequence: if the prompt says order/from first to last but not order of appearance, verify whether it asks physical scene/path order.
- Physical/environmental: verify scene context, object affordances, support, containment, terrain, and environmental state.
- Depth/proximity: use occlusion, contact points, floor/support, and latest co-visible span.
- Social: track who can see/hear/interact with whom, not merely who is on screen.
- Vertical spatial: separate physical above/below from depth overlap and image y-position artifacts.
- Viewpoint/visibility: distinguish camera visibility from a character's line of sight.
- If the evidence still does not support the draft, change the answer.

Return exactly one JSON object:
{{
  "question_id": "{question.get('question_id')}",
  "answer_choice": "one listed option letter",
  "confidence": 0.0,
  "final_rationale": "short answer-grounded rationale",
  "changed_from_draft": false,
  "audit_notes": ["brief falsification checks performed"]
}}"""
