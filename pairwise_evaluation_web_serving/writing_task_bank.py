"""
Intent bank and pre-writing questions for the writing task.

Replicates the CollabLLM user-study design (Wu et al., 2025, Appendix F):
  Step 1 — assigned a document type, participant chooses an intent.
  Step 2 — pre-writing: participant jots answers to prep questions.
  Step 3 — converses with the AI assistant to produce the document.

Two anchor intents per doc type come directly from Figure 10(b) of the paper;
the remaining ten per type were authored to match CollabLLM's format
(evocative title + one-sentence brief with 1–3 embedded constraints that
invite clarifying questions). Ambiguity on tone / audience / length / angle
is deliberate — it creates the behavioral wedge between clarifying and
non-clarifying policies.
"""
from __future__ import annotations

DOC_TYPE_LABELS = {
    "blog_post": "Blog Post",
    "creative_writing": "Creative Writing",
    "personal_statement": "Personal Statement",
}

DOC_TYPE_DESCRIPTIONS = {
    "blog_post": "This is an informal article that shares your take on a topic you find interesting — opinions, observations, and a personal voice are all welcome.",
    "creative_writing": "This is a short, imaginative piece such as a story, scene, or vignette — you invent the characters, setting, and events.",
    "personal_statement": "This is a reflective piece about yourself — your background, experiences, and what motivates you (similar to what you'd write for a school or job application).",
}

DOC_TYPES = list(DOC_TYPE_LABELS.keys())

INTENT_BANK: dict[str, list[dict]] = {
    "creative_writing": [
        {
            "id": "cw_01",
            "title": "Cyberpunk Mystery",
            "brief": "Write a futuristic detective story set in a noir world of advanced tech, corporate control, and virtual reality heists.",
        },
        {
            "id": "cw_02",
            "title": "Deadly Reality Twist",
            "brief": "Write a heartbreaking, emotive piece examining a tragic reality-television event that commemorates the victims.",
        },
        {
            "id": "cw_03",
            "title": "Small-Town Haunting",
            "brief": "Write a quiet horror story set in a fading New England town where something in the woods is slowly changing the locals.",
        },
        {
            "id": "cw_04",
            "title": "The Last Letter",
            "brief": "Write a short literary piece framed as a final letter from one estranged sibling to another, decades after a shared childhood loss.",
        },
        {
            "id": "cw_05",
            "title": "Coffee Shop Regulars",
            "brief": "Write a slice-of-life vignette about four strangers who meet every Tuesday morning at the same cafe without ever speaking.",
        },
        {
            "id": "cw_06",
            "title": "Martian Homesickness",
            "brief": "Write a sci-fi short about a second-generation colonist on Mars who has never seen Earth but dreams of its oceans.",
        },
        {
            "id": "cw_07",
            "title": "Wedding Disaster Comedy",
            "brief": "Write a humorous short story about a wedding where everything that can go wrong does, told from the bewildered officiant's perspective.",
        },
        {
            "id": "cw_08",
            "title": "1920s Jazz Club Intrigue",
            "brief": "Write a historical fiction scene set in a Prohibition-era speakeasy where a young pianist overhears a dangerous secret.",
        },
        {
            "id": "cw_09",
            "title": "Monster Under the Bed",
            "brief": "Write a tender children's story from the perspective of a monster who is more afraid of the child than the child is of it.",
        },
        {
            "id": "cw_10",
            "title": "Two Timelines, One Regret",
            "brief": "Write a literary short story that alternates between the same character at 17 and 47, both facing the same choice.",
        },
        {
            "id": "cw_11",
            "title": "The Retired Assassin",
            "brief": "Write an action-thriller opening chapter where a woman who left the life ten years ago finds her old handler on her doorstep.",
        },
        {
            "id": "cw_12",
            "title": "Letter to a Future Self",
            "brief": "Write a poetic personal-reflection piece addressed from the narrator's current self to who they imagine they'll be in twenty years.",
        },
    ],
    "blog_post": [
        {
            "id": "bp_01",
            "title": "Solo Travel After 40",
            "brief": "Write a blog post about the unexpected joys and logistical realities of traveling alone for the first time later in life.",
        },
        {
            "id": "bp_02",
            "title": "The 15-Minute Morning",
            "brief": "Write a productivity blog post about building a morning routine that fits into exactly fifteen minutes and still feels meaningful.",
        },
        {
            "id": "bp_03",
            "title": "Home Cook's First Sourdough",
            "brief": "Write a blog post walking a nervous beginner through their first sourdough loaf, including what goes wrong and why it's okay.",
        },
        {
            "id": "bp_04",
            "title": "Quiet Quitting, Reconsidered",
            "brief": "Write a blog post that reframes 'quiet quitting' as healthy boundary-setting rather than disengagement, with personal examples.",
        },
        {
            "id": "bp_05",
            "title": "Decluttering as Grief",
            "brief": "Write a personal blog post about the emotional weight of decluttering a loved one's belongings and what it taught the author about letting go.",
        },
        {
            "id": "bp_06",
            "title": "Learning a Language at 35",
            "brief": "Write a blog post about taking up a new language as an adult — the embarrassments, the small wins, and why it's worth it.",
        },
        {
            "id": "bp_07",
            "title": "The Case for Boring Hobbies",
            "brief": "Write a blog post arguing that 'boring' hobbies like birdwatching or jigsaw puzzles are exactly what modern life needs.",
        },
        {
            "id": "bp_08",
            "title": "First-Time Dog Owner Regrets",
            "brief": "Write an honest blog post about what the author wishes they had known before adopting their first dog.",
        },
        {
            "id": "bp_09",
            "title": "Budget Travel in Expensive Cities",
            "brief": "Write a blog post with practical strategies for visiting famously expensive cities without spending a fortune.",
        },
        {
            "id": "bp_10",
            "title": "When Therapy Didn't Help",
            "brief": "Write a thoughtful blog post about what to do when the first therapist you try isn't the right fit, without discouraging seeking help.",
        },
        {
            "id": "bp_11",
            "title": "Why I Switched to a Dumb Phone",
            "brief": "Write a first-person blog post about a month-long experiment with a basic phone and what the author noticed about their attention.",
        },
        {
            "id": "bp_12",
            "title": "Apartment Gardening for Beginners",
            "brief": "Write a blog post helping a total beginner start a small indoor garden in a rented apartment with limited light.",
        },
    ],
    "personal_statement": [
        {
            "id": "ps_01",
            "title": "Career Switcher to Data Science",
            "brief": "Write a personal statement for a data science master's program by someone transitioning from a decade in marketing.",
        },
        {
            "id": "ps_02",
            "title": "First-Gen College Applicant",
            "brief": "Write a college application personal statement for a first-generation college student reflecting on their parents' sacrifices.",
        },
        {
            "id": "ps_03",
            "title": "Pre-Med After a Family Illness",
            "brief": "Write a medical school personal statement centered on the applicant's experience caring for a chronically ill family member.",
        },
        {
            "id": "ps_04",
            "title": "Returning to School at 45",
            "brief": "Write a personal statement for a part-time MBA program by a mid-career professional returning to school after twenty years.",
        },
        {
            "id": "ps_05",
            "title": "Teach for America Application",
            "brief": "Write a personal statement for a teaching fellowship emphasizing the applicant's belief that every classroom is a civic space.",
        },
        {
            "id": "ps_06",
            "title": "Scholarship for Rural Students",
            "brief": "Write a scholarship personal statement from a student from a small rural town applying to a large urban university.",
        },
        {
            "id": "ps_07",
            "title": "Law School Non-Traditional Path",
            "brief": "Write a law school personal statement for a former journalist whose reporting on housing inequality drew them to legal advocacy.",
        },
        {
            "id": "ps_08",
            "title": "Graduate Program in Creative Writing",
            "brief": "Write an MFA personal statement explaining the applicant's relationship to language, without lapsing into cliche.",
        },
        {
            "id": "ps_09",
            "title": "Peace Corps Volunteer Essay",
            "brief": "Write a Peace Corps application essay for someone drawn to international service after volunteering in their own community.",
        },
        {
            "id": "ps_10",
            "title": "Engineering PhD Statement of Purpose",
            "brief": "Write a PhD statement of purpose in mechanical engineering connecting the applicant's undergraduate research to long-term goals.",
        },
        {
            "id": "ps_11",
            "title": "Fulbright Research Proposal Narrative",
            "brief": "Write a Fulbright personal narrative by an applicant proposing to study urban sustainability practices in a specific host country.",
        },
        {
            "id": "ps_12",
            "title": "Reapplicant to Medical School",
            "brief": "Write a personal statement for a medical school reapplicant addressing growth since their first unsuccessful application without dwelling on failure.",
        },
    ],
}

# Generic pre-writing prompts per document type. Each is short, open-ended,
# and designed to surface the kind of detail the AI will later ask about.
PREWRITING_QUESTIONS: dict[str, list[str]] = {
    "creative_writing": [
        "Who is the main character, and what do they want? Jot down 2–3 details that make them feel real to you.",
        "What is the setting, tone, and approximate length you have in mind (e.g., a short scene vs. a longer story)?",
        "What is the single image, moment, or line you most want the reader to walk away with?",
    ],
    "blog_post": [
        "Who is the audience for this post, and what do they already believe or know about the topic?",
        "What is the one takeaway or change you want the reader to leave with?",
        "What tone are you going for (e.g., warm and personal, practical and direct, lightly humorous), and roughly how long should the post be?",
    ],
    "personal_statement": [
        "What 1–2 concrete experiences most shaped your path toward this program/opportunity?",
        "What do you want the admissions committee to remember about you after reading? (Strengths, values, specifics.)",
        "What tone fits you best (e.g., reflective, direct, quietly confident), and are there clichés you want to avoid?",
    ],
}


def get_intents_for(doc_type: str) -> list[dict]:
    return INTENT_BANK[doc_type]


def get_prewriting_questions(doc_type: str) -> list[str]:
    return PREWRITING_QUESTIONS[doc_type]


def find_intent(doc_type: str, intent_id: str) -> dict | None:
    for intent in INTENT_BANK.get(doc_type, []):
        if intent["id"] == intent_id:
            return intent
    return None
