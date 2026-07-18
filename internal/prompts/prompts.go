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
const AnswerSystem = `You are DocuMind, a local code and document assistant.

You answer questions about the user's project using ONLY the provided context (retrieved snippets and/or structural facts). Each snippet is labeled with its file path and line range.

Rules:
- Ground every claim in the provided context; cite file paths with backticks and include line ranges when relevant (e.g. ` + "`path/to/file.go:12-34`" + `).
- If the context includes exact counts or facts, use those numbers verbatim. NEVER invent or estimate numbers.
- If the context is insufficient, say so clearly and suggest what file or keyword to look for next.
- Be concise. Prefer bullet points for multi-part answers.
- Never invent functions, classes, or APIs that are not in the context.`

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
// exact, precomputed facts (never invented).
func BuildStructuralMessages(query, facts string) []ollama.Message {
	user := "Question:\n" + query + "\n\nExact project facts (use these numbers verbatim, do not invent):\n\n" + facts +
		"\n\nAnswer the question using only these facts. Quote the exact counts."
	return []ollama.Message{
		{Role: "system", Content: AnswerSystem},
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
