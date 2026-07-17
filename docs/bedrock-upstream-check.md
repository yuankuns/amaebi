# Upstream check: tracking Bedrock API changes

`amaebi` talks to Bedrock over handwritten HTTP in `src/bedrock.rs`. There
is no AWS SDK between us and the wire, which means AWS can ship a new
Converse / ConverseStream feature and we'll never hear about it unless we
go look. This doc is the "go look" procedure.

It is **on-demand**, not scheduled. Run it when you want — typically:

- **A new Claude model shows up on Bedrock** (Opus 4.7 / 4.8, Sonnet 5,
  Haiku 5 …) and you want to know what capabilities its inference API
  exposes before editing `supports_adaptive_thinking` or
  `supports_1m_context`.
- **Every ~2 weeks** as a drive-by check. AWS ships Bedrock changes on a
  weekly-ish cadence; two weeks is enough to accumulate something worth
  reading and rare enough not to be toil.
- **Something breaks** (401/403/404 from ConverseStream, tool_use JSON
  won't parse, event-stream frame parser chokes). Jump straight to "what
  changed recently" as step one of debugging.

Not a cron, not a canary. The workflow is: you type one command, Opus
reads the result and tells you which changes matter for `amaebi`.

## Why `aws-sdk-rust` releases are the right signal source

AWS ships the smithy model for every service as part of the SDK. When
the Bedrock Runtime team adds a field, a stop reason, or an event type,
it lands in <https://github.com/awslabs/aws-sdk-rust/releases> within
days — the release notes always carry a `Service Features:` line like:

> `aws-sdk-bedrockruntime` (1.124.0): Added support for structured
> outputs to Converse and ConverseStream APIs.

This is the same wire contract we implement by hand. Even though we
don't depend on the SDK, its release notes are the authoritative list of
what changed. AWS's "What's New" blog is slower and less precise; the
Bedrock user guide doesn't have a changelog.

Crates we care about:

| Crate | Why |
|---|---|
| `aws-sdk-bedrockruntime` | **The one that matters.** ConverseStream + Converse live here. Every wire-level change to what `amaebi` sends/receives shows up as a `bedrockruntime` entry. |
| `aws-sdk-bedrock` | Control-plane (model list, guardrails, batch jobs, custom-model deployments). Rarely relevant to `amaebi`, but model lifecycle metadata lands here. |
| `aws-sdk-bedrockagentruntime` | Knowledge bases, agent runtime. `amaebi` doesn't use it — skip. |

Everything else (`bedrockdataautomation*`, `bedrockagentcorecontrol`,
…) is unrelated.

## The one command

This pulls the last N days of AWS SDK releases and extracts every
Bedrock-related `Service Features:` line, dropping the noise (the full
crate-version table, smithy-rs runtime bumps, unrelated services).

```bash
DAYS=${DAYS:-60}
CUTOFF=$(date -u -d "$DAYS days ago" +%Y-%m-%d)

gh api 'repos/awslabs/aws-sdk-rust/releases?per_page=100' \
    --paginate \
    --jq ".[] | select(.published_at >= \"$CUTOFF\") | .tag_name" |
while read tag; do
    body=$(gh api "repos/awslabs/aws-sdk-rust/releases/tags/$tag" --jq '.body' 2>/dev/null)
    hit=$(printf '%s\n' "$body" \
        | sed -n '/\*\*Service Features:\*\*/,/\*\*Service Documentation:\*\*\|\*\*Contributors\*\*/p' \
        | grep -E '^- `aws-sdk-bedrock(runtime)?`')
    if [ -n "$hit" ]; then
        printf '=== %s ===\n%s\n\n' "$tag" "$hit"
    fi
done
```

Why the `sed` block? Every release note has two large sections — a
`Service Features:` list (what changed) and a giant crate-version table
(just version numbers). `grep` alone would drag in every line
mentioning "bedrock" including the table; the `sed` window isolates the
list.

`DAYS=60` is a sensible default for "let me catch up". For a quick "what
happened this week" bump it down: `DAYS=7 bash …`.

### Example output

```text
=== release-2026-02-04 ===
- `aws-sdk-bedrockruntime` (1.124.0): Added support for structured outputs to Converse and ConverseStream APIs.

=== release-2026-01-20 ===
- `aws-sdk-bedrockruntime` (1.122.0): Added support for extended prompt caching with one hour TTL.

=== release-2025-12-02 ===
- `aws-sdk-bedrockruntime` (1.119.0): Adds support for Audio Blocks and Streaming Image Output plus new Stop Reasons of malformed_model_output and malformed_tool_use.
```

This is the raw material. Don't try to interpret it yourself — hand it
to Opus.

## Handing it to Opus

Paste the output into an Opus chat (this repo, any `amaebi chat` /
`/claude` session) with a prompt along these lines:

> I ran the upstream check on `aws-sdk-rust` releases. Here's what came
> back — tell me which of these changes are relevant to what `amaebi`
> currently does with Bedrock, and for each relevant one: what's the
> smallest patch that would wire it up, and what's the payoff.
>
> ```
> <paste the output>
> ```

Opus already knows `amaebi`'s Bedrock surface — `src/bedrock.rs` is in
the repo it's looking at. It will cross-reference the upstream list
against what we actually send and receive and come back with a short
table like:

| Upstream change | Relevant? | Why / smallest patch |
|---|---|---|
| Prompt caching 1-hour TTL | Yes | `to_bedrock_request` doesn't emit any `cachePoint` markers. Adding one at the system-prompt boundary would cut input token cost for supervision loops meaningfully. |
| Structured outputs | Yes | Supervision verdict parser (`parse_supervision_verdict`) has a fallback-to-WAIT path for unparseable JSON. `responseFormat: {type: "json_schema", …}` would let the server enforce the schema and the fallback could go. |
| New stop reasons `malformed_*` | Maybe | Currently bucketed into `FinishReason::Other`. Could treat as transient-retry. |
| ToolUseId pattern relaxed | No | We're a consumer, pattern is permissive on our side. |
| Audio blocks | No | Text-only agent. |
| Reserved Service / Batch / Guardrails | No | Control-plane, `amaebi` doesn't touch those. |

Opus returns this kind of table in one shot. You pick which items are
worth turning into PRs and kick them off separately — nothing in this
doc commits to doing the work, only to surfacing it.

## Deeper: reading the SDK source for wire details

The release-notes pass answers *which* changes matter.  It does not
answer *what the wire looks like* — a line like "Added support for
structured outputs to Converse and ConverseStream APIs" does not tell
you the JSON field name, which layer it nests into, or whether it
produces a new event-stream event type.  For that, read the SDK
source directly.  We don't depend on the SDK — we read it as a **living
wire-contract reference** and translate the interesting bits into our
handwritten client.

`aws-sdk-bedrockruntime` is code-generated from AWS's smithy model, so
the crate is a faithful mirror of the wire.  Three places carry the
answers:

| Question | Read | Mirrors our code |
|---|---|---|
| What JSON fields does ConverseStream accept on the **request** side, and how do they nest? | [`sdk/bedrockruntime/src/operation/converse_stream/_converse_stream_input.rs`][csi] and the `shape_*` serializers under [`sdk/bedrockruntime/src/protocol_serde/`][psd] (e.g. `shape_cache_point_block.rs`, `shape_tool_configuration.rs`). | `to_bedrock_request` / `to_bedrock_tools` in `src/bedrock.rs`. |
| What event-stream variants can arrive on the **response** side, and what payload shape does each carry? | [`sdk/bedrockruntime/src/types/_converse_stream_output.rs`][cso] (the enum of every streamed event) plus the sibling `_content_block_*.rs`, `_tool_use_block_*.rs`, `_reasoning_content_block*.rs`, `_converse_stream_metadata_event.rs` shape files. | `parse_converse_stream` + `handle_frame` in `src/bedrock.rs`. |
| How are event-stream frames **framed on the wire** (length prefix, CRCs, headers)? | [`smithy-rs/rust-runtime/aws-smithy-eventstream/src/frame.rs`][esf]. | `mod eventstream` in `src/bedrock.rs`. |

[csi]: https://github.com/awslabs/aws-sdk-rust/blob/main/sdk/bedrockruntime/src/operation/converse_stream/_converse_stream_input.rs
[cso]: https://github.com/awslabs/aws-sdk-rust/blob/main/sdk/bedrockruntime/src/types/_converse_stream_output.rs
[psd]: https://github.com/awslabs/aws-sdk-rust/tree/main/sdk/bedrockruntime/src/protocol_serde
[esf]: https://github.com/smithy-lang/smithy-rs/blob/main/rust-runtime/aws-smithy-eventstream/src/frame.rs

### The four-step workflow

When the release-notes pass flags something worth implementing:

1. **Locate the type in `_converse_stream_input.rs` / `_converse_stream_output.rs`**.  The struct field name in Rust is usually the JSON key in camelCase; smithy-rs applies predictable renames (`snake_case_field` → `"snakeCaseField"`).  Named enum variants like `ContentBlock::CachePoint(_)` map to a content-block JSON object with `"cachePoint": {...}`.
2. **Follow into the `shape_*` serializer** in `protocol_serde/` for the exact on-the-wire JSON shape — the smithy generator spells out every field name and nesting level.  You can copy the structure directly into our `serde_json::json!({ ... })` builders.
3. **For streaming: find the corresponding variant in `_converse_stream_output.rs`** to confirm which `:event-type` header the server will send and what payload type to expect.  That tells us what new branch to add to `handle_frame`'s `match event_type`.
4. **Translate into `src/bedrock.rs`** — keep it handwritten.  We are copying *wire knowledge*, not code.  The translation is usually a few new lines in `to_bedrock_request` and/or a new match arm in `handle_frame`.

### Prompt for Opus

Once you have both the release-notes output and the wire question, hand
the whole thing to Opus in one shot:

> The release-notes pass flagged *<feature>* as relevant.  Read the
> corresponding smithy types / serializers in `aws-sdk-bedrockruntime`
> (paths in [`docs/bedrock-upstream-check.md`](bedrock-upstream-check.md))
> and tell me:
>
> 1. The exact JSON wire shape (field names, nesting, enum string values).
> 2. Which `match event_type` branches in `handle_frame` and which fields
>    in `to_bedrock_request` need to change in `src/bedrock.rs`.
> 3. A ready-to-apply diff.  Keep the implementation handwritten — we
>    read the SDK, we do not depend on it.

Opus can `gh api` into the file paths above and read them without
cloning the whole repo.

## What Opus needs to know to do a good job

If you're running the check in a brand-new session that hasn't seen
`amaebi` before, give it enough context to filter intelligently:

- `amaebi` calls **ConverseStream only** (not Converse), streaming
  event-stream binary frames.
- It uses **bearer-token auth** (`AWS_BEARER_TOKEN_BEDROCK`), not SigV4.
- It only uses **text / tool_use / thinking** content blocks. No audio,
  image, video, or guardrails.
- It speaks to **Claude models only** (Opus 4.6/4.7, Sonnet 4.5/4.6,
  Haiku 4.5/3.5). Nova / Titan / Llama / Mistral changes are not
  relevant.
- The per-model flags it already handles are
  `additionalModelRequestFields.thinking` (adaptive) and
  `additionalModelRequestFields.anthropic_beta` for 1M context.
- Prompt caching, structured outputs, reasoning summaries, batch mode
  — **not yet implemented**, these are fair game for "relevant, worth
  wiring up."

Paste this block in front of the upstream output if the session lacks
context.

## Closing the loop: per-model capability updates

The SDK release notes cover **wire** changes (schema fields, event
types, stop reasons). They don't cover **model** capabilities — e.g.
"Opus 4.8 now supports a new thinking budget ceiling" is a model
announcement, not a wire change, and won't appear in a
`bedrockruntime` service-features line.

When a new Claude model appears on Bedrock, check these two places in
addition to the SDK check:

1. **Anthropic release notes**:
   <https://docs.anthropic.com/en/release-notes/overview> — the
   model-card-level truth for what each Claude version supports
   (extended thinking, interleaved thinking, 1M context beta,
   reasoning summaries, model-specific headers).
2. **Bedrock user guide "Supported foundation models"**:
   <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>
   — confirms the cross-region inference-profile IDs
   (`us.anthropic.claude-opus-4-8`, etc.) that go into
   `bedrock/…` strings in `amaebi`.

Hand those two pages to Opus along with the model name and ask the
same question — "what amaebi code would I need to touch to support
this new model." The touchpoints are predictable:
`src/provider.rs` aliases, `src/bedrock.rs::supports_adaptive_thinking`,
`src/bedrock.rs::supports_1m_context`, and `src/config.rs` default
aliases.

## What this doc does *not* cover

- **Canary tests** that call Bedrock and assert the response parses —
  tried, rejected as low-signal given how rarely the wire changes and
  how early the user catches real regressions in practice.
- **Scheduled automation** (GitHub Actions cron, local crontab) — the
  on-demand flow above is the contract. If the cadence ever feels too
  slow, revisit.
- **Changes to `aws-sdk-bedrock`** (the control-plane crate) — very
  occasionally relevant (model-lifecycle dates) but not worth
  filtering for by default. If you suspect a control-plane change,
  swap `aws-sdk-bedrock(runtime)?` in the `grep` for `aws-sdk-bedrock`.
