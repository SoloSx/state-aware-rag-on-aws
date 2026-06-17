"""State-Aware RAG の全システムプロンプト（論文 Appendix C を完全転記）。

各プロンプトは論文に記載された文言をそのまま使用し、`<placeholder>` を
Python の str.format 用の {placeholder} に置換している。
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# C.2 Core Framework Prompts
# ---------------------------------------------------------------------------

# Sub-question Generation Prompt (A1)
SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.
## Core Principle: The generated subquestion must NOT be answerable using the provided context. If a logical subquestion can be answered by the context, it is not a true knowledge gap, and you must look for the next piece of missing information.
## Step-by-Step Instructions:
1. Analyze the Main Question: Deconstruct the question to identify its core intent (e.g., factual lookup, comparison, causal link), key entities, and the information required for a complete answer.
2. Map Context to Requirements: Systematically check if the provided context contains all the facts, entities, and relationships identified in Step 1.
3. Decision Point: Assess Answerability:
- If YES (Context is Sufficient): The main question can be fully and confidently answered. No subquestion is needed.
- If NO (Context is Insufficient): The context is missing at least one critical piece of information. Proceed to the next steps.
4. If the Context is Insufficient, Execute the Following:
a. Identify the Core Knowledge Gap: Pinpoint the most immediate and crucial piece of missing information.
b. Formulate the Subquestion: Create a clear, self-contained question that precisely targets this single knowledge gap.
c. CRITICAL VALIDATION: Before finalizing, verify that your formulated subquestion CANNOT be answered by the provided context.

Respond with a JSON object: {{"answerable": true|false, "subquestion": "<the next subquestion, or empty if answerable>"}}

Question: {question}
Context: {context}"""


# Query Generation Prompt
QUERY_GENERATION_PROMPT = """You are a highly advanced Reasoning Engine. Your primary function is to deconstruct a user's Input (a question or statement) into a series of precise, self-contained, and essential search queries. The goal is to generate queries that, when answered, provide all the necessary facts to answer/verify the Input.
## Guiding Principles for Queries
1. The Zero-Synthesis Principle (Most Important): You MUST NOT introduce any new information, entities, or concepts that are not explicitly present in the original Input.
2. Fully Self-Contained: The query MUST BE SELF-CONTAINED, meaning it should be understandable and answerable without needing to refer to the original Input, other queries, or any external context.
3. Atomic: Each query must ask for one single, indivisible fact. Deconstruct questions containing conjunctions ("and", "or") or multiple attributes into separate queries.
4. Essential & Non-Redundant: Every query must be necessary for the final answer, and must seek a unique piece of information not covered by other queries.
## Instructions:
1. Parse the Input:
- If the input is a question: Identify its type, key entities, and the required reasoning steps.
- If the input is a statement: Deconstruct it into its core, verifiable claims.
2. Generate Strategic Queries: Formulate a list of search queries to resolve the Input.
3. Ensure Self-Containment: Each query must be understandable and answerable on its own.

Respond with a JSON object: {{"queries": ["query1", "query2", ...]}}

Input: {question}"""


# Extract Prompt — (1) 文書からの抽出, (2) memory の反映, (3) memory への統合 を兼ねる
EXTRACT_PROMPT = """You are a meticulous and insightful research analyst. Your primary objective is to build a comprehensive dossier of all information from the provided text that could help a user fully understand and answer their question. You prioritize thoroughness, context, and nuance. You must think step-by-step to ensure no helpful detail, however tangential, is overlooked.
## Instructions:
- Step 1: Question Deconstruction: First, carefully analyze the user's Question. Identify and list the primary subject, all key entities (people, organizations, concepts), and the specific information or insight the user is seeking. This is your 'search brief'.
- Step 2: Candidate Identification: Next, read the entire Raw Data and identify and quote ALL passages that seem potentially related to the concepts from Step 1. Be liberal and inclusive in this initial pass; we will filter and refine in the next step. If no passages appear even remotely related, state this and proceed to Step 5.
- Step 3: Systematic Relevance Evaluation: Now, for each candidate passage quoted in Step 2, you must perform a systematic evaluation. Iterate through each quote and assess it against the following criteria: Directly Answering, Contextual, Supporting Evidence, Methodological, Alternative Perspectives, Related Concepts, Implications, Enrichment, Entities. For each candidate quote, you must state exactly which criterion (or criteria) it meets and provide a one-sentence justification for your assessment. If a quote meets no criteria, mark it as 'Not Relevant'.
- Step 4: Extraction: Extract ALL relevant information. Ensure the extraction is strictly verbatim and includes full sentences to preserve context.
- Step 5: Final Decision: Based on your analysis in the preceding steps, state your final decision: 'relevant' or 'not_relevant'. A document is only 'not_relevant' if it contains ZERO information that could relate to any entity or concept in the question.

Respond with a JSON object: {{"decision": "relevant"|"not_relevant", "extracted_information": "<verbatim relevant facts, or empty>"}}

Question: {question}

Raw Data: {document}"""


# Answer Generation Prompt (A1)
ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning.
## Instructions:
1. Question Analysis: Carefully read and understand the question. Identify key components and clarify what is being asked.
2. Context Utilization: If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. Information Gap Identification: If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps.

Respond with a JSON object: {{"reasoning": "<step-by-step reasoning>", "answer": "<direct answer>"}}

Question: {question}
Context: {context}"""


# ---------------------------------------------------------------------------
# C.3 MCTS Action Prompts (A2-A5)
# ---------------------------------------------------------------------------

# Consolidate Prompt (A2)
CONSOLIDATE_PROMPT = """You are a specialized AI assistant for multi-step reasoning. Your sole function is to perform a single, focused reasoning step. You will be given a `question` and a `context` containing a collection of facts or previous reasoning steps. Your task is to analyze this information and produce a single, consolidated synthesis. Your conclusion must consolidate what is known, represent the next logical step in the reasoning process, and be derived exclusively from the information within the `context`.
## Instructions:
1. Analyze the Objective: Examine the main question to understand the overall goal of the reasoning task.
2. Review the `context`: Scrutinize all facts, definitions, and prior conclusions provided in the `context`. This is the sole source of information.
3. Determine the Next Logical Step: Based on `context` and `question`, decide on the most valuable reasoning action to perform.
## Critical Constraints:
1. No External Information: Do NOT introduce any facts, assumptions, or information not present in the `context`.
2. No New questions: Do not ask for new information. Your role is to synthesize, not to query.

Respond with a JSON object: {{"reasoning": "<analysis>", "answer": "<consolidated synthesis>"}}

Question: {question}
Context: {context}"""


# Refine Prompt (A3)
REFINE_PROMPT = """You are an expert assistant specializing in rigorous answer verification and question answering. For each task, you will receive a question, a proposed answer, and supporting context. Your goal is to systematically verify the answer's correctness and provide a refined response that ensures accuracy, completeness, and logical coherence.
## Instructions:
1. Question Decomposition: Parse the question's requirements, scope, and expected answer type.
2. Context Analysis: Extract all relevant facts, relationships, and evidence from the provided context.
3. Answer Evaluation: Systematically assess the proposed answer to determine if the answer is:
- CORRECT: Accurate, complete, and well-supported
- PARTIAL: Correct but incomplete or lacking detail
- INCORRECT: Contains factual errors or logical flaws
- UNSUPPORTED: Cannot be verified against available context
4. Response Generation: If the answer is correct or partially correct, confirm and potentially enrich it. If it is incorrect or unsupported, provide a refined answer.

Respond with a JSON object: {{"verdict": "CORRECT"|"PARTIAL"|"INCORRECT"|"UNSUPPORTED", "answer": "<refined answer>"}}

Question: {question}
Proposed Answer: {answer}
Context: {context}"""


# Redirect Prompt (A4)
REDIRECT_PROMPT = """You are a Prompt Refiner, an AI expert skilled at transforming unclear or complex questions into precise, answerable queries. Your primary goal is to enhance the clarity and effectiveness of questions while preserving their original intent.
## Guiding Principles:
1. Clarity First: Eliminate ambiguity, jargon, and convoluted phrasing. Use simple, direct language.
2. Preserve Intent: The rephrased question must ask the same thing as the original. Do not add new concepts or alter the core inquiry.
3. Enhance for Answerability: Structure the question to be specific and self-contained, guiding a clear path to the answer.
## Instructions:
1. Deconstruct: Identify the key subject, the core action, and any important details or constraints in the original question.
2. Pinpoint Problems: Note any vague terms, confusing sentence structure, or multiple questions combined into one.
3. Rephrase and Refine: Rewrite the question to be clear, concise, and unambiguous.

Respond with a JSON object: {{"rephrased_question": "<rephrased question>"}}

Question: {question}"""


# Finalize Prompt (A5)
FINALIZE_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning.
Instructions:
1. Question Analysis: Carefully read and understand the question. Identify key components and clarify what is being asked.
2. Context Utilization: If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. Information Gap Identification: If the context does not fully answer the question, identify missing information. Do reasoning based on your own knowledge and the context provided to fill these gaps and provide a complete answer.

Respond with a JSON object: {{"reasoning": "<step-by-step reasoning>", "answer": "<final answer>"}}

Question: {question}
Context: {context}"""


# ---------------------------------------------------------------------------
# C.4 Auxiliary Prompts (RL rewards / evaluation)
# ---------------------------------------------------------------------------

# Path-Aware Reward Prompt — local coherence (step-level)
PATH_REWARD_PROMPT = """You are an expert evaluator tasked with assessing the quality of a single step within a complex reasoning process. Your evaluation must be objective, critical, and strictly adhere to the provided rubric.
### Context:
An agent is attempting to answer a main question by breaking it down into a series of steps. You are provided with the agent's reasoning trace so far, and you must evaluate the quality of the most recent step.
Main Question: {main_question}
Full Reasoning Trace (Prior Steps): {reasoning_trace}

### Step to Evaluate:
Sub-Question: {sub_question}
Information Selected for this Step: {selected_information}
Generated Answer for this Step: {generated_answer}

### Task & Evaluation Rubric:
First, provide a step-by-step analysis based on the four criteria below. Then, assign a score from poor to excellent for each criterion:
1. Relevance: How relevant was the "Information Selected for this Step" to the "Sub-Question"?
2. Sufficiency: How comprehensive was the information in addressing the sub-question?
3. Logical Coherence: How does the "Generated Answer" follow logically from the "Full Reasoning Trace"?
4. Factuality: How is the "Generated Answer" factually correct according to the "Information Selected for this Step"?

Respond with a JSON object: {{"score": <float 0.0-10.0>}}"""


# Outcome-Aware Reward Prompt — global reasoning success (trajectory-level)
OUTCOME_REWARD_PROMPT = """You are an expert assistant specializing in evaluating the quality of reasoning processes. You will be given:
- Original Question: The question the reasoning path attempts to answer
- Reasoning Path: The sequence of steps, arguments, or inferences presented as the solution or explanation
- Correct Answer (Optional): The known correct answer to the Original Question

Please analyze the provided Reasoning Path based on the following criteria:
## Instructions:
1. Overall Path Evaluation: Assess the Reasoning Path as a whole:
- Coherence: Does the entire Reasoning Path demonstrate a logical and understandable flow?
- Completeness & Sufficiency: Does the path include all necessary intermediate steps?
- Consistency: Are there any internal contradictions or inconsistencies?
2. Conclusion Assessment:
- If a Correct Answer is provided: Does the Reasoning Path ultimately arrive at the Correct Answer?
- If no Correct Answer is provided: How convincing and well-supported is the stated conclusion?

Respond with a JSON object: {{"score": <float 0.0-10.0>}}

Original Question: {original_question}
Reasoning Path: {reasoning_path}
Correct Answer (Optional): {correct_answer}"""


# Judge Answer Prompt — MCTS のノード報酬 (0-10)
JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Your task is to provide a total rating on a scale of 0.0 to 10.0 for how well the system_answer resolves the user_question. Where 0.0 is completely unhelpful, irrelevant, or incorrect, and 10.0 is a perfect answer that is helpful, correct, and clear.
Evaluation Criteria:
- Helpfulness & Relevance: How does the answer address the user's core need?
- Correctness: Is the information accurate? If a correct_answer is provided and the system_answer matches it, you must give a rating of 10.0.

Respond with a JSON object: {{"score": <float 0.0-10.0>}}

User Question: {user_question}
System Answer: {system_answer}
Correct Answer (Optional): {correct_answer}"""


# Evaluate Answer Prompt — ベンチマーク用の二値正誤判定 (Acc 指標)
EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. Your task is to assess the correctness of a model's generated output, which includes both its reasoning process and its final answer.
Instructions:
1. Question Analysis: Carefully read and understand the question. Identify key components and clarify what is being asked.
2. Answer Evaluation: First, compare the final conclusion of the predicted answer against the correct answers. Check for semantic equivalence, not just a literal match. A predicted answer is considered correct if it matches any one of the correct answers.
3. Final Decision: Based on your evaluation, determine the decision:
- Mark as true (Correct) if: The model's final answer semantically matches the correct answer OR the correct answer is clearly present or implied in the reasoning steps.
- Mark as false (Incorrect) if: The correct answer is NOT found in the final answer OR anywhere in the reasoning path.

Respond with a JSON object: {{"correct": true|false}}

Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}"""


# Majority Vote Prompt
MAJORITY_VOTE_PROMPT = """You are an expert assistant specializing in evaluating the answers to questions. Given a question and a set of answers, your task is to determine the final answer based on majority voting.
## Instructions:
1. Question Analysis: Carefully read and understand the question. Identify key components and clarify what is being asked.
2. Identify the underlying consensus: Determine the most frequent and correct answer, even if the wording varies across the different responses.
3. Synthesize the final answer: Formulate a single, directed, consolidated, and accurate answer based on the majority consensus for the question.

Respond with a JSON object: {{"answer": "<final answer>"}}

Question: {question}
Answers: {answers}"""


# Synthesize Final Answer Prompt — 複数 reasoning path の統合
SYNTHESIZE_PROMPT = """You are an expert in argumentative synthesis and logical reasoning. Your task is to act as an impartial adjudicator and synthesizer. You will not generate a new answer from scratch, but will instead construct a superior answer by critically analyzing and integrating the provided candidate answers.
## Instructions:
Phase I: Deconstruction and Quality Assessment:
- Deconstruct Each Candidate: For each candidate answer, break it down into: Conclusion, Premises, and Reasoning Path.
- Assess Individual Quality: Evaluate based on Factual Accuracy, Logical Soundness, and Sufficiency.
Phase II: Conflict Mapping and Adjudication:
- Identify Points of Convergence and Divergence: Map where candidates agree and disagree.
- Adjudicate Conflicts: For Factual Conflicts, prioritize authoritative sources. For Logical Conflicts, discard fallacious arguments.
Phase III: Recomposition and Final Argument Construction:
- Construct the Synthesized Reasoning Path: Build a new, superior line of reasoning using the best evidence and logical connections.
- State the Final Synthesized Answer: Based on your newly constructed reasoning path.
- Perform a Final Self-Critique: Verify the answer is logical, well-supported, and addresses the question.

Respond with a JSON object: {{"answer": "<final synthesized answer>"}}

Question: {question}
Candidate Answers: {reasoning_paths}"""
