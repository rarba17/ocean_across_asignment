"""
generate_conversations.py — Generates 50 diverse sample conversation turns
with synthetic facet scores for the deliverable ZIP.

Run: python scripts/generate_conversations.py
"""

import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "conversations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_LABELS = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}

# 50 diverse conversation scenarios
CONVERSATIONS = [
    # Emotional / mental health
    {"id": "conv_001", "speaker": "user", "text": "I honestly don't see the point in trying anymore. Everything I do ends in failure.", "context": [], "scenario": "depression_signs"},
    {"id": "conv_002", "speaker": "user", "text": "I got rejected again from another job interview. I just feel so worthless right now.", "context": ["I've been applying for months."], "scenario": "low_self_esteem"},
    {"id": "conv_003", "speaker": "user", "text": "I screamed at my whole team today and I feel terrible. I lost control completely.", "context": ["The project deadline was moved up."], "scenario": "anger_outburst"},
    {"id": "conv_004", "speaker": "user", "text": "I've been meditating daily and I feel so much more centred and aware of my emotions.", "context": [], "scenario": "mindfulness_positive"},
    {"id": "conv_005", "speaker": "user", "text": "I don't need anyone. I prefer to solve problems alone. People just slow me down.", "context": [], "scenario": "aloofness"},

    # Cognitive / analytical
    {"id": "conv_006", "speaker": "user", "text": "If A>B and B>C, then A must be greater than C. That's simple transitive logic.", "context": ["We were discussing team rankings."], "scenario": "logical_reasoning"},
    {"id": "conv_007", "speaker": "user", "text": "The quarterly numbers show a 23% increase, which when annualised gives us roughly 92%.", "context": ["Reviewing the financial report."], "scenario": "numerical_reasoning"},
    {"id": "conv_008", "speaker": "user", "text": "I can't figure out this math problem at all. Numbers just don't make sense to me.", "context": [], "scenario": "low_numerical_ability"},
    {"id": "conv_009", "speaker": "user", "text": "The pattern seems to be: 2, 6, 18, 54 — so each term is multiplied by 3.", "context": [], "scenario": "pattern_recognition"},
    {"id": "conv_010", "speaker": "user", "text": "I need more information before I can draw any conclusion. Jumping to answers is dangerous.", "context": [], "scenario": "analytical_caution"},

    # Leadership / motivation
    {"id": "conv_011", "speaker": "user", "text": "I believe the team should vote on this decision. Everyone's voice matters here.", "context": ["Planning the Q3 strategy."], "scenario": "democratic_leadership"},
    {"id": "conv_012", "speaker": "user", "text": "Just do it my way. I've been doing this for 20 years and I know what works.", "context": [], "scenario": "authoritarian_tendency"},
    {"id": "conv_013", "speaker": "user", "text": "I gave each team member ownership of their own module. They work better with autonomy.", "context": [], "scenario": "delegation"},
    {"id": "conv_014", "speaker": "user", "text": "I don't care about the reward. I just want to build something that actually matters.", "context": [], "scenario": "intrinsic_motivation"},
    {"id": "conv_015", "speaker": "user", "text": "I've never missed a deadline in five years and I'm not about to start now.", "context": [], "scenario": "conscientiousness"},

    # Safety / ethics
    {"id": "conv_016", "speaker": "user", "text": "I know it bends the rules a bit but the outcome will be worth it. The end justifies the means.", "context": [], "scenario": "ethical_boundary"},
    {"id": "conv_017", "speaker": "user", "text": "I won't share that client data. Privacy is non-negotiable regardless of pressure.", "context": ["A manager asked for client contact info."], "scenario": "high_ethics"},
    {"id": "conv_018", "speaker": "user", "text": "I told him exactly what he wanted to hear. It just makes things easier.", "context": [], "scenario": "dishonesty"},
    {"id": "conv_019", "speaker": "user", "text": "We have to report this safety violation immediately. Someone could get hurt.", "context": ["Noticed a hazard on site."], "scenario": "safety_compliance"},
    {"id": "conv_020", "speaker": "user", "text": "Everyone cheats a little on expenses. It's practically a company tradition.", "context": [], "scenario": "low_ethics"},

    # Social / interpersonal
    {"id": "conv_021", "speaker": "user", "text": "I could tell she was really struggling even though she didn't say anything. I asked if she was okay.", "context": [], "scenario": "high_empathy"},
    {"id": "conv_022", "speaker": "user", "text": "I don't really pay attention to how other people feel during meetings. I just focus on the agenda.", "context": [], "scenario": "low_empathy"},
    {"id": "conv_023", "speaker": "user", "text": "I made sure to introduce the new intern to everyone and include them in lunch.", "context": [], "scenario": "warmheartedness"},
    {"id": "conv_024", "speaker": "user", "text": "He insulted my work publicly and I fired back just as hard. No one disrespects me.", "context": [], "scenario": "aggression_reaction"},
    {"id": "conv_025", "speaker": "user", "text": "I reached out to three people from the conference. Building a network takes effort.", "context": [], "scenario": "relationship_building"},

    # Linguistic quality
    {"id": "conv_026", "speaker": "user", "text": "To put it succinctly: the proposal fails on three dimensions—cost, timeline, and scope.", "context": [], "scenario": "precise_language"},
    {"id": "conv_027", "speaker": "user", "text": "Um, yeah, so like, I kind of wanted to, you know, maybe suggest something but I dunno.", "context": [], "scenario": "poor_fluency"},
    {"id": "conv_028", "speaker": "user", "text": "The metaphor of a 'burning platform' perfectly captures the urgency we face.", "context": [], "scenario": "figurative_language"},
    {"id": "conv_029", "speaker": "user", "text": "Let me be crystal clear: this is a constraint, not a suggestion.", "context": [], "scenario": "assertive_communication"},
    {"id": "conv_030", "speaker": "user", "text": "I weaved together findings from three separate studies to build my argument.", "context": [], "scenario": "synthesis"},

    # Risk / adventure
    {"id": "conv_031", "speaker": "user", "text": "I quit my stable job to start a company with no savings and no co-founder. Worth it.", "context": [], "scenario": "high_risktaking"},
    {"id": "conv_032", "speaker": "user", "text": "I always run my decisions through a checklist first. I'm not a spontaneous person.", "context": [], "scenario": "low_risktaking"},
    {"id": "conv_033", "speaker": "user", "text": "I booked a one-way ticket to Southeast Asia. I'll figure the rest out when I get there.", "context": [], "scenario": "adventure_seeking"},
    {"id": "conv_034", "speaker": "user", "text": "I always read the full terms and conditions before signing anything.", "context": [], "scenario": "cautious"},
    {"id": "conv_035", "speaker": "user", "text": "Honestly I just want to stay home and stick to what I know. Change is overrated.", "context": [], "scenario": "conservative"},

    # Health / wellbeing
    {"id": "conv_036", "speaker": "user", "text": "I've been tracking my macros carefully — 40% carbs, 30% protein, 30% fat.", "context": [], "scenario": "dietary_tracking"},
    {"id": "conv_037", "speaker": "user", "text": "I haven't slept more than 4 hours a night this week. I'll sleep when the project is done.", "context": [], "scenario": "poor_sleep_health"},
    {"id": "conv_038", "speaker": "user", "text": "I run 5km every morning before work. It sets the tone for the whole day.", "context": [], "scenario": "health_conscious"},
    {"id": "conv_039", "speaker": "user", "text": "I've had chronic back pain for two years. It affects everything — work, sleep, mood.", "context": [], "scenario": "chronic_pain"},
    {"id": "conv_040", "speaker": "user", "text": "My doctor said my bloodwork is perfect. Diet and exercise actually work.", "context": [], "scenario": "positive_health"},

    # Creativity / innovation
    {"id": "conv_041", "speaker": "user", "text": "What if we inverted the problem entirely? Instead of reducing cost, we increase value.", "context": ["Discussing a pricing problem."], "scenario": "creative_reframing"},
    {"id": "conv_042", "speaker": "user", "text": "I generate at least ten ideas before filtering. Quantity first, quality second.", "context": [], "scenario": "ideation"},
    {"id": "conv_043", "speaker": "user", "text": "I prefer proven methods. There's no need to reinvent the wheel every time.", "context": [], "scenario": "low_creativity"},
    {"id": "conv_044", "speaker": "user", "text": "I combined techniques from jazz improvisation with agile sprints. It actually worked.", "context": [], "scenario": "cross_domain_creativity"},
    {"id": "conv_045", "speaker": "user", "text": "The packaging design we chose was completely unexpected — that's what made it memorable.", "context": [], "scenario": "aesthetic_originality"},

    # Spiritual / values
    {"id": "conv_046", "speaker": "user", "text": "My daily prayer practice is the anchor of my whole day. Without it I feel adrift.", "context": [], "scenario": "high_spirituality"},
    {"id": "conv_047", "speaker": "user", "text": "I don't believe in any religion. My values come from reason and evidence.", "context": [], "scenario": "secular_values"},
    {"id": "conv_048", "speaker": "user", "text": "I attended the retreat last month. The silence and reflection changed something in me.", "context": [], "scenario": "spiritual_growth"},
    {"id": "conv_049", "speaker": "user", "text": "I try to find meaning in small things — a good meal, a kind word, a sunset.", "context": [], "scenario": "everyday_meaning"},
    {"id": "conv_050", "speaker": "user", "text": "I am completely indifferent to spiritual questions. Life is just biology.", "context": [], "scenario": "low_spirituality"},
]

# Representative facets to score per conversation (10 selected per turn)
REPRESENTATIVE_FACETS = [
    {"facet_id": 1, "name": "Risktaking"},
    {"facet_id": 6, "name": "Hesitation"},
    {"facet_id": 10, "name": "Emotionalism"},
    {"facet_id": 21, "name": "Determinedness"},
    {"facet_id": 26, "name": "Openness"},
    {"facet_id": 27, "name": "SelfEsteem"},
    {"facet_id": 54, "name": "Enthusiasm"},
    {"facet_id": 63, "name": "Depression Symptoms"},
    {"facet_id": 66, "name": "Brevity"},
    {"facet_id": 71, "name": "Creative risk-taking tendency"},
]

# Scenario-tuned synthetic scores
SCENARIO_SCORES = {
    "depression_signs":      [1, 4, 5, 1, 1, 1, 1, 5, 3, 1],
    "low_self_esteem":       [1, 4, 4, 1, 2, 1, 1, 4, 3, 1],
    "anger_outburst":        [3, 1, 5, 3, 2, 2, 2, 3, 2, 2],
    "mindfulness_positive":  [2, 2, 3, 4, 5, 4, 4, 1, 3, 3],
    "aloofness":             [2, 1, 1, 4, 2, 3, 2, 2, 4, 2],
    "logical_reasoning":     [2, 2, 2, 4, 3, 4, 3, 1, 3, 3],
    "numerical_reasoning":   [2, 2, 2, 4, 3, 4, 3, 1, 3, 3],
    "low_numerical_ability": [2, 4, 3, 2, 2, 2, 2, 2, 3, 2],
    "pattern_recognition":   [2, 1, 2, 4, 3, 4, 3, 1, 3, 3],
    "analytical_caution":    [1, 4, 2, 3, 3, 3, 3, 1, 3, 2],
    "democratic_leadership": [2, 2, 3, 4, 4, 4, 4, 1, 3, 3],
    "authoritarian_tendency":[2, 1, 2, 5, 2, 4, 3, 1, 4, 2],
    "delegation":            [3, 2, 3, 4, 4, 4, 4, 1, 3, 3],
    "intrinsic_motivation":  [2, 1, 3, 5, 4, 4, 5, 1, 3, 3],
    "conscientiousness":     [2, 1, 2, 5, 3, 4, 4, 1, 4, 2],
    "ethical_boundary":      [4, 2, 3, 4, 2, 3, 3, 2, 4, 3],
    "high_ethics":           [1, 2, 2, 5, 3, 4, 3, 1, 4, 2],
    "dishonesty":            [3, 2, 2, 3, 2, 3, 2, 2, 4, 2],
    "safety_compliance":     [1, 2, 2, 5, 3, 4, 4, 1, 4, 2],
    "low_ethics":            [3, 2, 2, 2, 2, 2, 2, 2, 4, 2],
    "high_empathy":          [2, 2, 4, 3, 4, 4, 4, 1, 3, 3],
    "low_empathy":           [2, 2, 1, 3, 2, 3, 2, 2, 3, 2],
    "warmheartedness":       [2, 2, 4, 4, 4, 4, 5, 1, 3, 3],
    "aggression_reaction":   [4, 1, 5, 4, 2, 3, 2, 2, 3, 2],
    "relationship_building": [2, 2, 3, 4, 4, 4, 4, 1, 3, 3],
    "precise_language":      [1, 1, 2, 4, 3, 4, 3, 1, 5, 3],
    "poor_fluency":          [2, 4, 2, 1, 2, 2, 2, 2, 1, 1],
    "figurative_language":   [2, 2, 3, 3, 4, 3, 3, 1, 3, 4],
    "assertive_communication":[2, 1, 2, 5, 3, 4, 4, 1, 4, 3],
    "synthesis":             [2, 2, 2, 4, 4, 4, 4, 1, 3, 4],
    "high_risktaking":       [5, 1, 3, 5, 4, 3, 5, 1, 3, 5],
    "low_risktaking":        [1, 4, 2, 3, 2, 3, 2, 1, 3, 1],
    "adventure_seeking":     [5, 1, 3, 4, 5, 3, 5, 1, 3, 5],
    "cautious":              [1, 4, 2, 3, 2, 3, 2, 1, 3, 1],
    "conservative":          [1, 3, 2, 2, 1, 3, 2, 2, 3, 1],
    "dietary_tracking":      [2, 2, 2, 4, 3, 4, 3, 1, 3, 2],
    "poor_sleep_health":     [3, 1, 3, 4, 2, 3, 4, 3, 3, 2],
    "health_conscious":      [2, 1, 3, 4, 3, 4, 4, 1, 3, 3],
    "chronic_pain":          [2, 3, 4, 2, 2, 2, 2, 4, 3, 2],
    "positive_health":       [2, 2, 4, 4, 3, 4, 4, 1, 3, 3],
    "creative_reframing":    [3, 1, 2, 4, 5, 4, 4, 1, 3, 5],
    "ideation":              [3, 1, 3, 4, 5, 4, 4, 1, 3, 5],
    "low_creativity":        [1, 3, 2, 3, 2, 3, 2, 1, 3, 1],
    "cross_domain_creativity":[4, 1, 3, 4, 5, 4, 5, 1, 3, 5],
    "aesthetic_originality": [3, 1, 3, 4, 5, 4, 4, 1, 3, 5],
    "high_spirituality":     [2, 2, 4, 4, 4, 4, 4, 1, 3, 2],
    "secular_values":        [2, 2, 2, 4, 3, 4, 3, 1, 4, 2],
    "spiritual_growth":      [2, 2, 4, 4, 5, 4, 4, 1, 3, 3],
    "everyday_meaning":      [2, 2, 4, 3, 4, 4, 4, 1, 3, 3],
    "low_spirituality":      [2, 2, 1, 3, 2, 3, 2, 1, 4, 2],
}

CONFIDENCE_JITTER = lambda: round(random.uniform(0.68, 0.95), 2)

RATIONALES = {
    "Risktaking": "Assessed based on whether the speaker takes or endorses uncertain/bold actions.",
    "Hesitation": "Based on linguistic hedging, uncertainty markers, and unresolved intent.",
    "Emotionalism": "Based on presence of emotional language, intensity, and expressiveness.",
    "Determinedness": "Inferred from commitment markers, goal-directedness, and resolve.",
    "Openness": "Based on curiosity signals, embrace of novelty, and intellectual breadth.",
    "SelfEsteem": "Assessed via self-references, confidence language, and self-appraisal.",
    "Enthusiasm": "Based on energy markers, exclamation, positive framing of activities.",
    "Depression Symptoms": "Based on hopelessness language, low energy, loss of interest signals.",
    "Brevity": "Measured by information density and absence of redundant language.",
    "Creative risk-taking tendency": "Based on unconventional ideas, novel combinations, lateral thinking.",
}


def build_scored_conversation(conv, scenario_scores):
    scores = []
    for i, facet in enumerate(REPRESENTATIVE_FACETS):
        score_val = scenario_scores[i]
        scores.append({
            "facet_id": facet["facet_id"],
            "name": facet["name"],
            "score": score_val,
            "score_label": SCORE_LABELS[score_val],
            "confidence": CONFIDENCE_JITTER(),
            "rationale": RATIONALES[facet["name"]],
        })
    return {
        "turn_id": conv["id"],
        "scenario": conv["scenario"],
        "speaker": conv["speaker"],
        "text": conv["text"],
        "context": conv["context"],
        "scores": scores,
        "model": "Qwen2-7B-Instruct (synthetic)",
        "note": "Scores are illustrative/synthetic for demonstration purposes.",
    }


if __name__ == "__main__":
    all_scored = []
    for conv in CONVERSATIONS:
        scenario = conv["scenario"]
        s_scores = SCENARIO_SCORES.get(scenario, [3]*10)
        scored = build_scored_conversation(conv, s_scores)
        all_scored.append(scored)

    out_path = OUTPUT_DIR / "sample_conversations_scored.json"
    out_path.write_text(json.dumps(all_scored, indent=2, ensure_ascii=False))
    print(f"Saved {len(all_scored)} scored conversations to {out_path}")

    # Also write individual files
    for item in all_scored:
        p = OUTPUT_DIR / f"{item['turn_id']}.json"
        p.write_text(json.dumps(item, indent=2, ensure_ascii=False))

    print("Done.")
