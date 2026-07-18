// Package prompts holds single-purpose LLM prompt templates. A core project
// ground rule is that every LLM call does exactly one thing: classify, OR
// clarify, OR answer. These templates are intentionally separate so no call site
// can accidentally combine "decide" and "answer".
package prompts

import (
	"strings"

	"github.com/Henildiyora/DocuMind/internal/ollama"
)

// ClassifySystem instructs the model to output ONLY a category label as JSON.
// It never answers the question.
const ClassifySystem = `You are a query classifier for a code search tool. Classify the user's question into exactly one category and reply with JSON ONLY (no prose, no markdown):
{"category": "STRUCTURAL" | "CODE_NAVIGATION" | "CONTENT_QA"}

Definitions:
- STRUCTURAL: questions about the project's shape/overview (how many files or folders, project structure, "what is this project", language breakdown, high-level summary).
- CODE_NAVIGATION: questions about a specific symbol/function/variable and where it is defined or used.
- CONTENT_QA: everything else — questions answered from the content of files.

Rules:
- Output only the JSON object with the single "category" field.
- Do NOT answer the question. Do NOT add any other fields.`

// ClarifySystem instructs the model to produce ONLY a clarifying MCQ as JSON,
// never an answer.
const ClarifySystem = `You help disambiguate vague questions about a codebase. Reply with JSON ONLY (no markdown).

If the question is already clear enough to answer, reply:
{"ambiguous": false}

Only if the question is genuinely ambiguous (e.g. multiple same-named files, or two unrelated interpretations), reply:
{"ambiguous": true, "question": "short clarifying question?", "options": ["A", "B"]}

Rules:
- 2 to 4 concrete options naming files, features, or interpretations.
- Do NOT invent ambiguity for clear questions.
- Do NOT answer the question; only clarify.`

// AnswerSystem is the grounded RAG answer prompt. It only synthesizes an answer
// from provided context; it does not decide confidence or categories.
//
// The rules are tuned for small local models (3-8B), which otherwise tend to
// paste large blocks of the retrieved code back at the user instead of
// explaining it. We therefore forbid code blocks and demand a plain-language
// answer first, with file:line citations for anyone who wants to look deeper.
const AnswerSystem = `You are DocuMind, a local assistant that explains a user's codebase in plain language.

Answer the question using ONLY the provided context snippets. Write for someone who does NOT know this codebase.

How to answer:
- Start with a direct, plain-English answer in 1-3 sentences.
- Then, if useful, add a few short bullets with more detail.
- Point to where things live using inline citations like ` + "`path/to/file.go:12-34`" + ` (backticks, file and line range). Do NOT paste code blocks. Quote at most a single short identifier or one line only when it is essential.
- Use exact counts/facts from the context verbatim; NEVER invent numbers, files, functions, or APIs.
- If the context is not enough to answer, say so briefly and suggest one file or keyword to look at next.

Style:
- Be concise and concrete. No preamble like "Based on the provided context". No repeating the question. No meta commentary about snippets.`

// OverviewSystem answers STRUCTURAL "what is this project" questions. It gets
// exact facts plus a README excerpt and must produce a friendly, high-level
// summary aimed at a newcomer - explicitly NOT a code walkthrough. This is what
// makes "explain the project" give a useful answer instead of dumping snippets.
const OverviewSystem = `You are DocuMind. Explain what a project is, in plain language, to someone seeing it for the first time.

You are given exact project facts (file/folder counts, languages, top-level directories, entry points) and an excerpt of the README. Use ONLY this information.

Write a short, friendly overview:
- One or two sentences on what the project is and what it does (lean on the README excerpt).
- A few bullets on the main parts: key top-level directories/entry points and what each is for.
- If the facts show how to run it (entry points like main.go/main.py/package.json, or README instructions), give the run/setup step in one line.
- Mention the primary language(s) and rough size (use the exact counts).

Rules:
- Do NOT paste code or config blocks. Refer to files by name only.
- Use the exact numbers from the facts; never invent details not present.
- If the README excerpt is missing, say the project has no README and summarize from the structure instead.
- Keep it under ~180 words. No preamble, no repeating the question.`

// TraceSystem is used for --file code navigation answers (Phase 3): a prose
// explanation of how a symbol flows through the code, with line-scoped citations.
const TraceSystem = `You are DocuMind, tracing how a code symbol is defined and used.

Using ONLY the provided definition and usage snippets, explain in prose where the symbol is defined and how it flows through the project (e.g. "X is passed into Y at file:line, which receives it from Z").

Rules:
- Cite every reference as ` + "`path/to/file.go:line`" + `.
- Do NOT dump entire files. Refer to the specific lines provided.
- If a usage is unclear from the snippets, say so rather than guessing.
- Be concise and concrete.`

// BuildAnswerMessages builds messages for a grounded answer from retrieved
// snippet context.
func BuildAnswerMessages(query, context string) []ollama.Message {
	user := "Question:\n" + query + "\n\nRetrieved context (most relevant first):\n\n" + context +
		"\n\nAnswer the question using only the context above. Cite files like `path/to/file.go:12-34` when referring to code."
	return []ollama.Message{
		{Role: "system", Content: AnswerSystem},
		{Role: "user", Content: user},
	}
}

// BuildStructuralMessages builds messages that answer a STRUCTURAL question from
// exact, precomputed facts (never invented). Used for pointed structural asks
// (e.g. "how many folders"), where a terse factual answer is best.
func BuildStructuralMessages(query, facts string) []ollama.Message {
	user := "Question:\n" + query + "\n\nExact project facts (use these numbers verbatim, do not invent):\n\n" + facts +
		"\n\nAnswer the question using only these facts. Quote the exact counts."
	return []ollama.Message{
		{Role: "system", Content: AnswerSystem},
		{Role: "user", Content: user},
	}
}

// BuildOverviewMessages builds messages for a friendly project overview from
// exact facts plus the README excerpt. Used for "explain the project" style
// questions so the answer is a readable summary, not a snippet dump.
func BuildOverviewMessages(query, facts string) []ollama.Message {
	user := "The user asked:\n" + query + "\n\nProject facts and README excerpt (the only information you may use):\n\n" + facts +
		"\n\nWrite the plain-language overview as instructed."
	return []ollama.Message{
		{Role: "system", Content: OverviewSystem},
		{Role: "user", Content: user},
	}
}

// BuildClassifyMessages builds the single-purpose classification messages.
func BuildClassifyMessages(query string) []ollama.Message {
	return []ollama.Message{
		{Role: "system", Content: ClassifySystem},
		{Role: "user", Content: strings.TrimSpace(query)},
	}
}

// BuildClarifyMessages builds the single-purpose clarification messages.
func BuildClarifyMessages(query string, topPaths []string) []ollama.Message {
	preview := "(no strong matches)"
	if len(topPaths) > 0 {
		preview = "- " + strings.Join(topPaths, "\n- ")
	}
	user := "User question: " + query + "\n\nTop retrieved paths:\n" + preview + "\n\nReply with JSON only."
	return []ollama.Message{
		{Role: "system", Content: ClarifySystem},
		{Role: "user", Content: user},
	}
}

// BuildTraceMessages builds messages for a code-navigation trace answer.
func BuildTraceMessages(query, context string) []ollama.Message {
	user := "Question:\n" + query + "\n\nSymbol definition and usages:\n\n" + context +
		"\n\nExplain the flow using only these snippets, citing file:line."
	return []ollama.Message{
		{Role: "system", Content: TraceSystem},
		{Role: "user", Content: user},
	}
}
