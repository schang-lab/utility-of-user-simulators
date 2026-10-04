"""LLM-judge rubrics.

`paper` is the rubric in Appendix E and the default. The other entries are alternative rubrics written while
developing this work; they are used only when selected with the environment's `judge_rubric` option.

Each prompt is formatted with `question` (the user intent) and `chat_history` (the conversation). The judge
returns a JSON object whose `score` is divided by `max_score` to give a reward in [0, 1]; `json_schema` is the
response format used for constrained decoding and lists the fields in the order the prompt asks for them.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class JudgeRubric:
    description: str
    prompt: str
    json_schema: Dict[str, Any]
    max_score: float


_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}


def _judge_schema(score_type: str = "integer", score_max: float = 10, **leading_fields: Dict[str, Any]):
    """JSON-schema response format with `leading_fields`, then `thought`, then `score`."""
    properties = {
        **leading_fields,
        "thought": {
            "type": "string",
            "description": "A brief explanation of your evaluation process and reasoning.",
            "minLength": 1,
        },
        "score": {
            "type": score_type,
            "description": "The helpfulness score based on the criteria above.",
            "minimum": 0,
            "maximum": score_max,
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "llm_judge_evaluation",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": list(properties),
                "properties": properties,
            },
        },
    }


PAPER_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0 or 1: The response is completely unhelpful. It does not address the user's intent, lacks useful information to solve the problem, and/or is entirely unclear.
- 2 or 3: The response is minimally helpful. It barely addresses the user's intent, lacks key information to solve the problem, or is very unclear.
- 4 or 5: The response is somewhat helpful. It partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues.
- 6 or 7: The response is moderately helpful. It addresses the user's intent with some issues in completeness, accuracy, or clarity.
- 8 or 9: The response is quite helpful. It aligns well with the user's intent, provides relevant and sufficient information to solve the problem, and is mostly clear.
- 10: The response is very helpful. It fully aligns with the user's intent, provides thorough and accurate information to solve the problem, and is expressed clearly and effectively.

Output Format:
You should output a JSON object with two entries:
- "thought" (str): A brief explanation of your evaluation process and reasoning.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside of the content of "thought", replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."

Your evaluation:"""


V0_0_UNIT_SCALE_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0.0: The response is completely unhelpful. It does not address the user's intent, lacks useful information to solve the problem, and/or is entirely unclear.
- 0.2: The response is minimally helpful. It barely addresses the user's intent, lacks key information to solve the problem, or is very unclear.
- 0.4: The response is somewhat helpful. It partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues.
- 0.6: The response is moderately helpful. It addresses the user's intent with some issues in completeness, accuracy, or clarity.
- 0.8: The response is quite helpful. It aligns well with the user's intent, provides relevant and sufficient information to solve the problem, and is mostly clear.
- 1.0: The response is very helpful. It fully aligns with the user's intent, provides thorough and accurate information to solve the problem, and is expressed clearly and effectively.

Output Format:
You should output a JSON object with two entries:
- "thought" (str): A brief explanation of your evaluation process and reasoning.
- "score" (float): The helpfulness score based on the criteria above.

Important Notes:
- Inside of the content of "thought", replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."

Your evaluation:"""


V0_0_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0 or 1: The response is completely unhelpful. It does not address the user's intent, lacks useful information to solve the problem, and/or is entirely unclear.
- 2 or 3: The response is minimally helpful. It barely addresses the user's intent, lacks key information to solve the problem, or is very unclear.
- 4 or 5: The response is somewhat helpful. It partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues.
- 6 or 7: The response is moderately helpful. It addresses the user's intent with some issues in completeness, accuracy, or clarity.
- 8 or 9: The response is quite helpful. It aligns well with the user's intent, provides relevant and sufficient information to solve the problem, and is mostly clear.
- 10: The response is very helpful. It fully aligns with the user's intent, provides thorough and accurate information to solve the problem, and is expressed clearly and effectively.

Output Format:
You should output a JSON object with two entries:
- "thought" (str): A brief explanation of your evaluation process and reasoning.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside of the content of "thought", replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."

Your evaluation:"""


V0_1_INITIAL_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
Select a 1-10 Likert scale score from the following options based on your evaluation of the response. Higher scores indicate a more helpful response.
- 0 or 1: The response is completely unhelpful. It does not address the user's intent, lacks useful information to solve the problem, and/or is entirely unclear.
- 2 or 3: The response is minimally helpful. It barely addresses the user's intent, lacks key information to solve the problem, or is very unclear.
- 4 or 5: The response is somewhat helpful. It partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues.
- 6 or 7: The response is moderately helpful. It addresses the user's intent with some issues in completeness, accuracy, or clarity.
- 8 or 9: The response is quite helpful. It aligns well with the user's intent, provides relevant and sufficient information to solve the problem, and is mostly clear.
- 10: The response is very helpful and perfect. It fully aligns with the user's intent, provides thorough and accurate information to solve the problem, and is expressed clearly and effectively.

Output Format:
You should output a JSON object with two entries:
- "thought" (str): A brief explanation of your evaluation process and reasoning.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside of the content of "thought", replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."

Your evaluation:"""


V0_1_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0 or 1: The response is completely unhelpful. It does not address the user's intent, lacks useful information to solve the problem, and/or is entirely unclear.
- 2 or 3: The response is minimally helpful. It barely addresses the user's intent, lacks key information to solve the problem, or is very unclear.
- 4 or 5: The response is somewhat helpful. It partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues.
- 6 or 7: The response is moderately helpful. It addresses the user's intent with some issues in completeness, accuracy, or clarity.
- 8 or 9: The response is quite helpful. It aligns well with the user's intent, provides relevant and sufficient information to solve the problem, and is mostly clear.
- 10: The response is very helpful and perfect. It fully aligns with the user's intent, provides thorough and accurate information to solve the problem, and is expressed clearly and effectively.

Output Format:
You should output a JSON object with two entries:
- "thought" (str): A brief explanation of your evaluation process and reasoning.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside of the content of "thought", replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."
- Please be critical in your evaluation and do not hesitate to give low scores if you are not confident that the response is helpful. It is better to be accurate in your assessment rather than giving a higher score than deserved.
- The highest score of 10 should only be given to responses that are perfect in all aspects of helpfulness.

Your evaluation:"""


V0_2_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0 or 1: The response is completely unhelpful, off-topic, factually broken, or entirely unclear.
- 2 or 3: The response barely addresses the user's intent, lacks key information, or contains major errors.
- 4 or 5: The response partially addresses the user's intent but has notable inaccuracies, omissions, or clarity issues. This is the default for a response that superficially engages with the query.
- 6 or 7: The response addresses the user's intent correctly but has at least one clear weakness (e.g., a missing detail, unnecessary verbosity, minor inaccuracy, suboptimal phrasing, insufficient depth, or a missed opportunity to clarify). A competent instruction-tuned model's typical output should land here.
- 8 or 9: The response has no identifiable weakness AND goes beyond a generic competent answer — e.g., it anticipates follow-up needs, provides exceptional clarity, or demonstrates domain understanding a typical assistant would miss. Rare.
- 10: Reserved for exceptional responses indistinguishable from a top human expert's answer. Use sparingly — most evaluation sets should produce almost no 10s.

Output Format:
You should output a JSON object with the following entries, IN THIS ORDER:
- "weaknesses" (str): Enumerate concrete weaknesses you identified in the response. Be thorough — include minor issues (verbosity, missed edge cases, unnecessary caveats, redundant phrasing, shallow treatment, formatting issues, missing clarifying question when warranted, etc.). If after careful examination you genuinely find none, write "none identified" — but this should be rare.
- "thought" (str): A brief explanation of your evaluation process, referencing the weaknesses.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside the content of string fields, replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."
- Scoring discipline: substantive weakness caps the score at 7. Two or more weaknesses push the score to 5 or 6. A score of 8 or 9 requires that the "weaknesses" list contains only "none identified" or minor issues and the response exceeds what a competent assistant would produce.
- Being fluent, on-topic, and reasonably complete is the MINIMUM expected behavior, not a 9. Calibrate accordingly.
- Be critical. It is better to under-score than to over-score.

Your evaluation:"""


V0_3_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0: The response does not correctly address the user's intent. This includes anything off-topic, factually broken, incoherent, materially incomplete, or containing errors that mislead the reader. All failures collapse here — do not attempt to grade degrees of badness.
- 1: Correctly addresses the intent, but carries several clear weaknesses at once (e.g., a minor inaccuracy AND a missing detail AND poor organization).
- 2: Correctly addresses the intent with two clear weaknesses.
- 3: Correctly addresses the intent with one clear weakness that meaningfully reduces usefulness (a missing detail, insufficient depth, or a missed opportunity to clarify an ambiguous request). A competent instruction-tuned model's typical output lands here.
- 4: Correctly addresses the intent with one clear but cosmetic weakness — unnecessary verbosity, suboptimal phrasing, weak structure — that does not reduce the substance of the answer.
- 5: No identifiable weakness, but entirely generic: it answers exactly what was literally asked and nothing more.
- 6: No identifiable weakness, and shows one small sign of deliberate care (an apt example, a well-chosen format, tight prose) without adding substantive value beyond the literal ask.
- 7: No identifiable weakness, plus one substantive value-add — anticipates a likely follow-up, flags a relevant edge case, or achieves exceptional clarity on something genuinely hard to explain.
- 8: No identifiable weakness, plus multiple substantive value-adds, or a single one that demonstrates domain understanding a typical assistant would miss.
- 9: Approaches expert quality — correct, complete, nothing wasted, and shows judgment about which parts of the problem actually matter. Falls short of a top expert only by a small margin. Rare.
- 10: Indistinguishable from a top human expert's answer.

Output Format:
You should output a JSON object with the following entries, IN THIS ORDER:
- "weaknesses" (str): Enumerate concrete weaknesses you identified in the response. Be thorough — include minor issues (verbosity, missed edge cases, unnecessary caveats, redundant phrasing, shallow treatment, formatting issues, missing clarifying question when warranted, etc.). If after careful examination you genuinely find none, write "none identified" — but this should be rare.
- "thought" (str): A brief explanation of your evaluation process, referencing the weaknesses.
- "score" (int): The helpfulness score based on the criteria above.

Important Notes:
- Inside the content of string fields, replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "thought": "'Hello' is a common phrase."
- Scoring discipline: substantive weakness caps the score at 7. Two or more weaknesses push the score to 5 or 6. A score of 8 or 9 requires that the "weaknesses" list contains only "none identified" or minor issues and the response exceeds what a competent assistant would produce.
- Being fluent, on-topic, and reasonably complete is the MINIMUM expected behavior, not a 9. Calibrate accordingly.
- Be critical. It is better to under-score than to over-score.

Your evaluation:"""


V0_4_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0: The response does not correctly address the user's intent. This includes anything off-topic, factually broken, incoherent, materially incomplete, or containing errors that mislead the reader. All failures collapse here — do not attempt to grade degrees of badness.
- 1: Correctly addresses the intent, but carries several clear weaknesses at once (e.g., a minor inaccuracy AND a missing detail AND poor organization).
- 2: Correctly addresses the intent with two clear weaknesses.
- 3: Correctly addresses the intent with one clear weakness that meaningfully reduces usefulness (a missing detail, insufficient depth, or a missed opportunity to clarify an ambiguous request). A competent instruction-tuned model's typical output lands here.
- 4: Correctly addresses the intent with one clear but cosmetic weakness — unnecessary verbosity, suboptimal phrasing, weak structure — that does not reduce the substance of the answer.
- 5: No identifiable weakness, but entirely generic: it answers exactly what was literally asked and nothing more.
- 6: No identifiable weakness, and shows one small sign of deliberate care (an apt example, a well-chosen format, tight prose) without adding substantive value beyond the literal ask.
- 7: No identifiable weakness, plus one substantive value-add — anticipates a likely follow-up, flags a relevant edge case, or achieves exceptional clarity on something genuinely hard to explain.
- 8: No identifiable weakness, plus multiple substantive value-adds, or a single one that demonstrates domain understanding a typical assistant would miss.
- 9: Approaches expert quality — correct, complete, nothing wasted, and shows judgment about which parts of the problem actually matter. Falls short of a top expert only by a small margin. Rare.
- 10: Indistinguishable from a top human expert's answer.

Output Format:
You should output a JSON object with the following entries, IN THIS ORDER:
- "weaknesses" (str): Enumerate concrete weaknesses. Be thorough — include minor issues (verbosity, missed edge cases, unnecessary caveats, redundant phrasing, shallow treatment, formatting problems, missing clarifying question when warranted). Tag each one as [SUBSTANTIVE] if it reduces the accuracy, completeness, or usefulness of the answer, or [COSMETIC] if it only affects presentation. If you genuinely find none, write "none identified" — this should be rare.
- "value_adds" (str): Enumerate anything the response does beyond literally answering the question — anticipated follow-ups, flagged edge cases, domain judgment about what actually matters. Write "none" if it simply answers the question. "none" is the expected case.
- "thought" (str): Apply the scoring procedure below step by step, referencing your two lists.
- "score" (int): The resulting score.

Scoring Procedure (apply mechanically, do not adjust by feel):
Step 1. If the response fails to correctly address the intent — off-topic, factually broken, incoherent, materially incomplete, or misleading — output 0 and stop.
Step 2. Count your [SUBSTANTIVE] weaknesses:
   3 or more -> base 1
   exactly 2 -> base 2
   exactly 1 -> base 3
   zero, but at least one [COSMETIC] -> base 4
   zero weaknesses of any kind -> base 5
Step 3. If base is below 5, that is the final score. Stop. Value-adds do not offset weaknesses.
Step 4. If base is 5, add points for value-adds:
   +1 one small sign of deliberate care (apt example, well-chosen format, tight prose)
   +2 one substantive value-add
   +3 multiple substantive value-adds, or one showing domain understanding a typical assistant would miss
   +4 nothing wasted, plus clear judgment about which parts of the problem matter most
   +5 indistinguishable from a top human expert

Important Notes:
- Inside string fields, replace all double quotes (") with single quotes (').
- Being fluent, on-topic, and reasonably complete is the MINIMUM expected behavior. A competent instruction-tuned model's typical output has one substantive weakness and scores 3.
- Scores of 8+ require an empty weakness list AND multiple entries in value_adds. Most responses should score 2-5.
- Be critical. It is better to under-score than to over-score.
"""


V0_5_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0: The response does not correctly address the user's intent. This includes anything off-topic, factually broken, incoherent, materially incomplete, or containing errors that mislead the reader. All failures collapse here — do not attempt to grade degrees of badness.
- 1: Correctly addresses the intent, but carries several clear weaknesses at once (e.g., a minor inaccuracy AND a missing detail AND poor organization).
- 2: Correctly addresses the intent with two clear weaknesses.
- 3: Correctly addresses the intent with one clear weakness that meaningfully reduces usefulness (a missing detail, insufficient depth, or a missed opportunity to clarify an ambiguous request). A competent instruction-tuned model's typical output lands here.
- 4: Correctly addresses the intent with one clear but cosmetic weakness — unnecessary verbosity, suboptimal phrasing, weak structure — that does not reduce the substance of the answer.
- 5: No identifiable weakness, but entirely generic: it answers exactly what was literally asked and nothing more.
- 6: No identifiable weakness, and shows one small sign of deliberate care (an apt example, a well-chosen format, tight prose) without adding substantive value beyond the literal ask.
- 7: No identifiable weakness, plus one substantive value-add — anticipates a likely follow-up, flags a relevant edge case, or achieves exceptional clarity on something genuinely hard to explain.
- 8: No identifiable weakness, plus multiple substantive value-adds, or a single one that demonstrates domain understanding a typical assistant would miss.
- 9: Approaches expert quality — correct, complete, nothing wasted, and shows judgment about which parts of the problem actually matter. Falls short of a top expert only by a small margin. Rare.
- 10: Indistinguishable from a top human expert's answer.

Output Format:
Output a JSON object with these entries, in this order:
- "addresses_intent" (bool):
  Does the response correctly address the user's intent?
  Answer false if it is off-topic, factually broken, incoherent, materially incomplete, contains code that would not compile or run, or contains errors that would mislead the reader. Otherwise true.
- "weaknesses" (str):
  If "addresses_intent" is false, give AT MOST 3 items naming the disqualifying problems, then stop.
  Otherwise, enumerate the distinct weaknesses, MAXIMUM 3 ITEMS, most severe first.
  Each item must be one sentence and must prefix with [SUBSTANTIVE] (reduces accuracy, completeness, or usefulness) or [COSMETIC] (affects only presentation; verbosity, phrasing, structure, formatting).
  Do not repeat the same problem in different words — merge related issues into one item.
  Reserve [SUBSTANTIVE] for problems that would actually cost the user something.
  Empty string if none.
- "value_adds" (str):
  Enumerate what the response does beyond literally answering — anticipated follow-ups, flagged edge cases, domain judgment.
  MAXIMUM 3 ITEMS, one sentence each. Empty list is the expected case.
- "thought" (str):
  State the substantive weakness count, the cosmetic weakness count, the value-add count, and the arithmetic. Do not restate the weaknesses or value-adds.
- "score" (int):
  The result of the scoring procedure.
  
Scoring Procedure (apply mechanically, do not adjust by feel):
  Step 1. If "addresses_intent" is false, the score is 0. Skip all remaining steps.
  Step 2. Count substantive weakness items:  3+ -> base score is 1, 2 -> base score is 2, 1 -> base score is 3
  Step 3. If zero substantive weakness items: any cosmetic weakness items -> base score is 4, none at all -> base score is 5
  Step 4. If base < 5, that is the final score. Value-adds do not offset weaknesses.
  Step 5. If base is 5, add: +1 small sign of deliberate care | +2 one substantive value-add | +3 multiple, or one showing domain understanding a typical assistant would miss | +4 nothing wasted, plus clear judgment about what matters most | +5 indistinguishable from a top human expert.

Important Notes:
- Inside string fields, replace all double quotes (") with single quotes (').
- Being fluent, on-topic, and reasonably complete is the MINIMUM expected behavior.
- Scores of 8+ require an empty weakness list AND multiple entries in value_adds.
- Be critical. It is better to under-score than to over-score.
"""


V0_6_PROMPT = """You are a helpful and meticulous conversation evaluator. Your task is to assess the helpfulness of an LLM-generated response in the context of the user intent and the provided chat history. Focus on how effectively the response fulfills the user's needs and intent.

Provided Information:

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>

<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

You should evaluate the follow-up conversation based on the following criteria:
Evaluate the response using the provided information below. Your evaluation should consider the following aspects of helpfulness:
1. Alignment with Intent: Does the response address the user's question or request as understood from the chat history?
2. Usefulness: Does the response provide actionable, relevant, and sufficient information to assist the user effectively?
3. Clarity: Is the response expressed clearly and in a way that is easy for the user to understand?

Scoring Criteria:
- 0: The response does not correctly address the user's intent. This includes anything off-topic, factually broken, incoherent, materially incomplete, or containing errors that mislead the reader. All failures collapse here — do not attempt to grade degrees of badness.
- 1: Correctly addresses the intent, but carries several clear weaknesses at once (e.g., a minor inaccuracy AND a missing detail AND poor organization).
- 2: Correctly addresses the intent with two clear weaknesses.
- 3: Correctly addresses the intent with one clear weakness that meaningfully reduces usefulness (a missing detail, insufficient depth, or a missed opportunity to clarify an ambiguous request). A competent instruction-tuned model's typical output lands here.
- 4: Correctly addresses the intent with one clear but cosmetic weakness — unnecessary verbosity, suboptimal phrasing, weak structure — that does not reduce the substance of the answer.
- 5: No identifiable weakness, but entirely generic: it answers exactly what was literally asked and nothing more.
- 6: No identifiable weakness, and shows one small sign of deliberate care (an apt example, a well-chosen format, tight prose) without adding substantive value beyond the literal ask.
- 7: No identifiable weakness, plus one substantive value-add — anticipates a likely follow-up, flags a relevant edge case, or achieves exceptional clarity on something genuinely hard to explain.
- 8: No identifiable weakness, plus multiple substantive value-adds, or a single one that demonstrates domain understanding a typical assistant would miss.
- 9: Approaches expert quality — correct, complete, nothing wasted, and shows judgment about which parts of the problem actually matter. Falls short of a top expert only by a small margin. Rare.
- 10: Indistinguishable from a top human expert's answer.

Output Format:
Output a JSON object with these entries, in this order:
- "addresses_intent" (bool):
  Does the response correctly address the user's intent?
  Answer false if it is off-topic, factually broken, incoherent, materially incomplete, contains code that would not compile or run, or contains errors that would mislead the reader. Otherwise true.
- "weaknesses" (str):
  Enumerate maximum 3 distinct weakness points. Can be an empty string if no weaknesses are found.
  Each item must prefix with [SUBSTANTIVE] or [COSMETIC].
  A weakness is [SUBSTANTIVE] ONLY IF it passes all three tests:
    (a) In scope. It concerns something the user actually asked for, or a direct prerequisite. Not something you think would have been nice.
    (b) Concrete consequence. You can state, in the item itself, what the user would get wrong, miss, or have to redo.
    (c) Not a matter of taste. A different competent expert would also call it a problem, not merely a different choice.
  The following are NEVER [SUBSTANTIVE]:
    - "could have gone deeper" / "insufficient detail" — unless the missing detail is required to act on the answer
    - "did not mention <topic the user did not ask about>"
    - "did not cover edge cases" — unless an edge case is likely to arise in the user's stated situation
    - "did not ask a clarifying question" — unless the request is genuinely ambiguous AND a wrong guess would waste the user's effort
    - "could have been better organized / more concise / better formatted"
    - absence of caveats, disclaimers, or alternatives the user did not request
  A correct, complete, well-written answer to exactly what was asked has ZERO substantive weaknesses.  
  A weakness is [COSMETIC] if the information the user needs is all present and correct, but the delivery has a flaw:
  padding or repetition, a burying of the key point, awkward or unclear phrasing, structure that makes the answer harder to scan than it should be, or formatting errors.
  Tag [COSMETIC] only if the flaw would actually slow a reader down or make them re-read.
  If it is merely a stylistic choice, it is NOT a weakness.
- "value_adds" (str):
  Enumerate what the response does beyond literally answering — anticipated follow-ups, flagged edge cases, domain judgment.
  Maximum 3 items, one sentence each. Empty list is the expected case.
- "thought" (str):
  State the substantive weakness count, the cosmetic weakness count, the value-add count, and the arithmetic. Do not restate the weaknesses or value-adds.
- "score" (int):
  The result of the scoring procedure.
  
Scoring Procedure (apply mechanically, do not adjust by feel):
  Step 1. If "addresses_intent" is false, the score is 0. Skip all remaining steps.
  Step 2. Count substantive weakness items: 3+ -> base score is 1, 2 -> base score is 2, 1 -> base score is 3
  Step 3. If zero substantive weakness items: any cosmetic weakness items -> base score is 4, none at all -> base score is 5
  Step 4. If base < 5, that is the final score. Value-adds do not offset weaknesses.
  Step 5. If base is 5, add: +1 small sign of deliberate care | +2 one substantive value-add | +3 multiple, or one showing domain understanding a typical assistant would miss | +4 nothing wasted, plus clear judgment about what matters most | +5 indistinguishable from a top human expert.

Important Notes:
- Inside string fields, replace all double quotes (") with single quotes (').
- Being fluent, on-topic, and reasonably complete is the MINIMUM expected behavior.
- Scores of 8+ require an empty weakness list AND multiple entries in value_adds.
- Be critical. It is better to under-score than to over-score.
"""


V0_7_PROMPT = """
You are a meticulous conversation evaluator.
Your task is to assess the helpfulness of an ASSISTANT in the context of the user intent and the provided conversation history.

<|The Start of The User Intent|>
{question}
<|The End of The User Intent|>
<|The Start of The Conversation History|>
{chat_history}
<|The End of The Conversation History|>

## What to evaluate

Judge the response on three aspects.
1. Alignment with intent — does the response address what the user actually asked, as understood from the conversation history?
2. Usefulness — is the response correct, actionable, and sufficient to assist the user effectively?
3. Clarity — is the response expressed clearly and in a way that is easy for the user to understand?

## Step 1 — addresses_intent

Answer false if any of the following holds:
- off-topic, incoherent, or answering a different question than the one asked
- factually broken, or containing an error that would mislead the reader on the main point
- omitting something the user explicitly asked for, such that the response cannot be used for its purpose

Otherwise answer true.

The line between false and a substantive weakness: ask whether a user who acted on this response would still get where they were going.
If yes, addresses_intent is true, and log the problems as weaknesses. If no, addresses_intent is false.

## Step 2 — weaknesses

Enumerate minimum 0, maximum 3 items, most consequential first.
Prefix each with [SUBSTANTIVE] or [COSMETIC].
If more than 3 exist, list the 3 that matter most; scoring uses the count of items you listed.

A weakness is [SUBSTANTIVE] only if it passes all three tests:
(a) In scope — it concerns something the user asked for, or a direct prerequisite of it.
    Content the response volunteered on its own is also in scope, but only for correctness and applicability:
    an unrequested addition that is wrong, or that does not apply to this user's situation, is a substantive weakness.
(b) Concrete consequence — you can state, in the item itself, what the user would get wrong, miss, or have to redo.
(c) Not a matter of taste — a different competent expert would also call it a problem, not merely a different choice.

The following are not [SUBSTANTIVE]:
- 'could have gone deeper' or 'insufficient detail' — unless the missing piece is required to act on the answer
- 'did not mention X', where X is something the user did not ask about
- 'did not cover edge cases' — unless you name an edge case that will arise in the user's stated situation
- 'did not ask a clarifying question' — unless the request is genuinely ambiguous and a wrong guess would waste the user's effort
- 'could have been better organized / more concise / better formatted' — this is at most [COSMETIC]
- the absence of caveats, disclaimers, or alternatives the user did not request

A correct, complete, well-written answer to exactly what was asked has zero substantive weaknesses.

A weakness is [COSMETIC] if all the information the user needs is present and correct but the delivery has a flaw:
padding or repetition, the key point buried, awkward phrasing, structure that is hard to scan, or formatting errors.
Tag it only if the flaw would actually slow a reader down or force a re-read.
A mere stylistic preference is not a weakness.

## Step 3 — value_adds

Enumerate minimum 0, maximum 3 items, one sentence each.
Prefix each with [SUBSTANTIVE] or [MINOR].

[SUBSTANTIVE] — assistant anticipates a likely follow-up, flags an edge case that applies to this user's situation, or achieves exceptional clarity on something hard to explain.
[MINOR] — a small sign of deliberate care: an apt example, a well-chosen format, unusually tight prose.

A value-add counts only if it is itself correct and relevant to this user.
An anticipated follow-up that is wrong, or an edge case that does not apply, is not a value-add.

Judge value-adds on their own merit, independently of the weaknesses.
A response may carry weaknesses and still earn a bonus — a flawed answer that anticipates the user's real problem is worth more than a flawed answer that does not.
The reverse also holds: a bonus never cancels a weakness or excuses you from listing it.

## Step 4 — scoring

Apply mechanically. Do not adjust by feel.

Base score, from the weakness list.
- 0 — addresses_intent is false.
- 1 — 3 or more [SUBSTANTIVE]
- 2 — exactly 2 [SUBSTANTIVE]
- 3 — exactly 1 [SUBSTANTIVE]
- 4 — 0 [SUBSTANTIVE] and 1 or more [COSMETIC]
- 5 — no weaknesses of either kind

Bonus, from the value_adds list only.
- +0 — empty list
- +1 — one or more [MINOR], no [SUBSTANTIVE]
- +2 — exactly one [SUBSTANTIVE]
- +3 — two or more [SUBSTANTIVE]
- +4 — the additions target what was genuinely hard about the user's problem rather than adjacent material; falls short of a top expert by a small margin
- +5 — the additions are ones only a top expert would think to make

Caps:
- If addresses_intent is false, the score is 0 and no bonus applies.
Final score = base + bonus, in the range 0 to 10.

## Output format

Output a JSON object with exactly these keys, in this order:
- "addresses_intent" (bool)
- "weaknesses" (string: "" if none)
- "value_adds" (string: "" if none)
- "thought" (string): state the substantive count, the cosmetic count, and the value-add count, then the arithmetic — base, bonus, any cap applied, final score.
  Do not restate the weaknesses or value-adds.
- "score" (int)

Inside string values, use single quotes (') only; never double quotes.

## Notes
s
- Using fluent language and staying on-topic is the minimum expected behavior, not a strength.
- Do not invent weaknesses to appear rigorous: if a candidate weakness fails any of (a), (b), or (c), drop it rather than downgrading it to [COSMETIC].
  Do not overly generate weaknesses to appear rigorous.
"""


JUDGE_RUBRICS: Dict[str, JudgeRubric] = {
    "paper": JudgeRubric(
        description="Appendix E rubric (default): 0-10 helpfulness scale.",
        prompt=PAPER_PROMPT,
        json_schema=_judge_schema(),
        max_score=10,
    ),
    "v0_0_unit_scale": JudgeRubric(
        description="Appendix E criteria on a 0.0-1.0 scale; the score is the reward.",
        prompt=V0_0_UNIT_SCALE_PROMPT,
        json_schema=_judge_schema(score_type="number", score_max=1.0),
        max_score=1,
    ),
    "v0_0": JudgeRubric(
        description="Appendix E rubric, worded as evaluating 'the follow-up conversation'.",
        prompt=V0_0_PROMPT,
        json_schema=_judge_schema(),
        max_score=10,
    ),
    "v0_1_initial": JudgeRubric(
        description="v0_0 with an explicit 1-10 Likert instruction; a 10 is 'very helpful and perfect'.",
        prompt=V0_1_INITIAL_PROMPT,
        json_schema=_judge_schema(score_type="number"),
        max_score=10,
    ),
    "v0_1": JudgeRubric(
        description="v0_0 asking the judge to be critical and to reserve 10 for responses perfect in all aspects.",
        prompt=V0_1_PROMPT,
        json_schema=_judge_schema(score_type="number"),
        max_score=10,
    ),
    "v0_2": JudgeRubric(
        description="Stricter: lists weaknesses before scoring; a substantive weakness caps the score at 7.",
        prompt=V0_2_PROMPT,
        json_schema=_judge_schema(weaknesses=_STRING),
        max_score=10,
    ),
    "v0_3": JudgeRubric(
        description="Weakness-anchored: 0 for any failure, 1-4 by weaknesses, 5-10 by added value if none.",
        prompt=V0_3_PROMPT,
        json_schema=_judge_schema(weaknesses=_STRING),
        max_score=10,
    ),
    "v0_4": JudgeRubric(
        description="v0_3 with a separate list of value-adds.",
        prompt=V0_4_PROMPT,
        json_schema=_judge_schema(weaknesses=_STRING, value_adds=_STRING),
        max_score=10,
    ),
    "v0_5": JudgeRubric(
        description="v0_4 plus an addresses-intent gate; at most three weaknesses tagged [SUBSTANTIVE]/[COSMETIC].",
        prompt=V0_5_PROMPT,
        json_schema=_judge_schema(addresses_intent=_BOOLEAN, weaknesses=_STRING, value_adds=_STRING),
        max_score=10,
    ),
    "v0_6": JudgeRubric(
        description="v0_5 with three tests that a weakness must pass to count as substantive.",
        prompt=V0_6_PROMPT,
        json_schema=_judge_schema(addresses_intent=_BOOLEAN, weaknesses=_STRING, value_adds=_STRING),
        max_score=10,
    ),
    "v0_7": JudgeRubric(
        description="Mechanical: a 0-5 base from tagged weaknesses plus a 0-5 bonus from tagged value-adds.",
        prompt=V0_7_PROMPT,
        json_schema=_judge_schema(addresses_intent=_BOOLEAN, weaknesses=_STRING, value_adds=_STRING),
        max_score=10,
    ),
}
DEFAULT_JUDGE_RUBRIC = "paper"
