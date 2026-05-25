# StudyAgent AI

> Final year undergraduate dissertation project (BSc Computer Science, 2025/26).

A multi-agent AI study assistant built on AWS Bedrock with the Strands Agents SDK. It ingests a student's own course materials (PDFs, notes) from S3, loads them in full into the orchestrator context, and routes work to specialist sub-agents that generate quizzes, flashcards, summaries, and cited research. A separate model-as-judge layer runs grounding checks against the source material on every response.

The project was submitted as a final year dissertation artefact and is published here as a portfolio piece. It demonstrates end-to-end AI engineering: agent orchestration, long-context document grounding, hallucination mitigation, persistent semantic memory, typed event streaming, and full observability on AWS managed infrastructure.

## What it shows

* **Multi-agent orchestration with model-driven routing.** A tiered model setup (Opus orchestrator, Sonnet specialists, Haiku verifier) where routing is driven by tool docstrings rather than imperative control flow. Cost and latency are optimised per tier.
* **Long-context document grounding instead of RAG.** The corpus fits comfortably inside the orchestrator's context window, so PDFs and notes are pulled from S3 and injected directly with source headers. The README documents the inflection point at which a vector store becomes worthwhile.
* **Model-as-judge verification.** A separate Haiku agent runs after every response, comparing the answer against the same source material the orchestrator saw and emitting a structured `GROUNDED / CONFIDENCE / NOTES` verdict. Skip rules avoid spending tokens on greetings and on already-structured outputs like flashcards.
* **Persistent semantic memory with three namespaces.** Facts, preferences, and per-session summaries are stored in AWS Bedrock AgentCore Memory, each with its own `top_k` and similarity threshold tuned for purpose. This is the only retrieval layer in the system.
* **Typed event streaming over SSE.** The Flask layer emits typed Server-Sent Events (`text`, `tool_call`, `flashcard_data`, `validation`, `analytics`, `done`, `error`) so the UI can render incrementally. The local orchestrator path (`src/agents/orchestrator.py`) uses a background-thread + thread-safe queue to bridge the SDK's async `stream_async` into Flask's sync generator.
* **Real cloud deployment.** Packaged as a Docker container, deployed to AWS Bedrock AgentCore Runtime, with CloudWatch logs, X-Ray traces, and the GenAI observability dashboard wired up.

## Tech stack

| Layer | Choice |
| --- | --- |
| Models | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 (via AWS Bedrock) |
| Agent framework | Strands Agents SDK |
| Runtime | AWS Bedrock AgentCore Runtime (serverless) |
| Memory | AWS Bedrock AgentCore Memory (STM + LTM, semantic) |
| Document store | S3 (raw PDF and text, loaded in full into context) |
| Analytics | DynamoDB |
| Search | Tavily API (academic-tuned web search, not document retrieval) |
| Frontend | Flask + Server-Sent Events |
| Observability | CloudWatch, X-Ray, AgentCore GenAI dashboard |

## Architecture

```
                           Student (browser)
                                  |
                                  v
                           Flask + SSE (app.py)
                                  |
                                  v
                  AWS Bedrock AgentCore Runtime
                                  |
                                  v
                  Orchestrator agent (Opus 4.6)
       _________________|_____________________________
      |          |           |          |             |
      v          v           v          v             v
   Quiz     Flashcard   Summariser   Search       Read URL
  (Sonnet)   (Sonnet)    (Sonnet)   (Sonnet)     (Sonnet)
                                  |
                                  v
                       Verification (Haiku 4.5)
                                  |
                                  v
                          Streaming response

  Sidecars: AgentCore Memory (semantic LTM), S3 (docs, loaded wholesale),
            DynamoDB (analytics), Secrets Manager (Tavily key),
            CloudWatch + X-Ray (traces)
```

## Specialist agents

* **Quiz** (Sonnet 4.6): multiple choice, true/false, short answer, difficulty controlled, with answer keys.
* **Flashcard** (Sonnet 4.6): chat-inline, Anki `.apkg`, or CSV export. A four-layer parser captures structured JSON / CSV through the orchestrator's paraphrasing layer.
* **Summariser** (Sonnet 4.6): bullet, paragraph, or hierarchical outline.
* **Web search** (Sonnet 4.6): Tavily-backed, quick or deep-research modes. Prioritises course materials over web results.
* **URL reader** (Sonnet 4.6): extracts and summarises a webpage the student pastes.
* **Verification** (Haiku 4.5): not exposed as a tool. Runs after every response so the orchestrator cannot skip it.

## Engineering decisions worth reading the code for

* `src/agents/orchestrator.py` (`_initialize_agent`, `_should_skip_validation`, the async-to-sync stream bridge).
* `src/tools/flashcard.py` (`_parse_flashcard_output`): the four-layer bypass that captures structured JSON / CSV before the orchestrator reformats it away.
* `src/tools/verification.py`: the grounding check, including the web-cross-check for general-knowledge claims and notes on the model-as-judge circularity limitation.
* `src/tools/_context.py`: why specialist tools share state through a module-level dict instead of through tool parameters (token economics).
* `src/utils/aws_resources.py` (`get_document_context`): how documents are loaded directly from S3 (no chunking, no embeddings) and stitched into the system prompt with source headers.
* `agentcore_app.py`: the `BedrockAgentCoreApp` entry point and how `invoke()` is wrapped for the cloud runtime.

## Honest limitations

* **No real retrieval over documents.** All loaded PDFs are injected directly into the prompt. This works because the corpus fits the context window. It does not scale past a few hundred thousand characters per session.
* **Model-as-judge shares failure modes with the generator.** Both Opus and Haiku are Claude models. They share training data and biases. A factually wrong answer that both models believe is correct will pass verification.
* **Cloud streaming is event-typed, not token-streamed.** The deployed AgentCore path returns a complete response, which `app.py` re-emits as typed SSE events. Only the local orchestrator path (`src/agents/orchestrator.py`) does true token streaming via `stream_async`.

## Environment variables

All AWS-specific values come from the environment. No account IDs, ARNs, or resource names are committed to the repo.

| Variable | Required | Purpose |
| --- | --- | --- |
| `AWS_PROFILE` | optional | boto3 profile name (omit to use the default credential chain or IAM role) |
| `AWS_REGION` | optional | AWS region (default `eu-west-1`, the only EU region with AgentCore at time of build) |
| `AGENTCORE_RUNTIME_ARN` | required by `app.py` | ARN of your deployed AgentCore Runtime agent |
| `AGENTCORE_MEMORY_ID` | required for memory | ID of your AgentCore Memory resource |
| `S3_BUCKET_NAME` | optional | Override the document storage bucket name |
| `ANALYTICS_TABLE_NAME` | optional | Override the DynamoDB analytics table name |
| `TAVILY_SECRET_NAME` | optional | Override the Secrets Manager secret name for the Tavily API key |
| `TAVILY_API_KEY` | optional | Alternative to Secrets Manager: set the key directly |

## Local setup

```bash
pip install -r requirements.txt
aws configure --profile my-profile
export AWS_PROFILE=my-profile
python setup_aws.py          # creates the S3 bucket and DynamoDB table
```

Create an AgentCore Memory resource via the `agentcore memory` CLI, then set `AGENTCORE_MEMORY_ID`.

## Deployment

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure --entrypoint agentcore_app.py --non-interactive
agentcore launch
```

`agentcore launch` builds the Docker image, pushes it to ECR, registers the agent, and deploys to the AgentCore Runtime. Set `AGENTCORE_RUNTIME_ARN` to the returned ARN, then:

```bash
python run.py
```

## Observability

Three layers, available out of the box once `observability.enabled: true` is set in `.bedrock_agentcore.yaml`:

* **CloudWatch Logs**: all `logging` output, searchable by `session_id`.
* **X-Ray traces**: nested spans across orchestrator, specialist sub-agents, and Bedrock API calls.
* **AgentCore GenAI dashboard**: token usage, invocation counts, latency p50/p95/p99, error rates by model, cost breakdown.

## Project structure

```
app.py                  Flask frontend, AgentCore Runtime client
agentcore_app.py        AgentCore Runtime entry point (deployed agent)
run.py                  Quick start script
setup_aws.py            S3 + DynamoDB bootstrap
src/
  config.py             Models, prompts, AWS settings (env-driven)
  agents/
    orchestrator.py     Multi-agent orchestrator (local dev path)
  tools/
    quiz.py             Quiz specialist
    flashcard.py        Flashcard specialist (four-layer parser)
    summarizer.py       Summary specialist
    search.py           Tavily web search
    read_url.py         URL extraction
    verification.py     Grounding check (model-as-judge)
    _context.py         Shared document context
  utils/
    aws_resources.py    S3 and DynamoDB management (document loader)
    analytics.py        DynamoDB interaction logger
templates/
  index.html            Chat UI
tests/
  test_studyagent.py    Integration test suite
```

## License

MIT
