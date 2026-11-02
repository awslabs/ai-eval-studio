# Mantle payload fixtures

Request and response bodies for the three Bedrock Mantle invocation APIs
(Responses, chat completions, Anthropic Messages), modelled on live captures.
Used by `tests/test_bedrock_mantle_mappers.py`, `tests/test_bedrock_mantle_model.py`,
`tests/test_bedrock_discovery.py` and `tests/test_bedrock_mantle_judge_cost.py`
to exercise the mappers, the adapter and the discovery probe without making real
HTTP calls.

## Source

Responses API:

- `responses_request.json` — the request body shape the adapter sends to
  `https://bedrock-mantle.{region}.api.aws/openai/v1/responses`. Verbatim
  from a live-tested capture.
- `responses_200_minimal.json` — the 200 response body for a completed
  Responses call (one reasoning block + one message block; usage block).
- `responses_incomplete_truncation.json` — the 200 body returned when the
  model hits `max_output_tokens` mid-thought; carries
  `status:'incomplete'` + `incomplete_details.reason:'max_output_tokens'`.
  Drives `MantleReasoningTruncated`.
- `responses_400_wrong_api.json` — the 400 `validation_error` body returned when
  a model is called on an API it does not support (here `openai.gpt-5.5` on
  `/v1/chat/completions`). This drives `MantleAPIRejected`; it is the
  load-bearing capability-probe signal consumed by the discovery layer.
- `responses_400_unsupported_temperature.json` — the 400 `unsupported_parameter`
  body a reasoning model returns when sent `temperature`. Drives the
  temperature self-correction.
- `responses_400_content_filter.json` — a 400 content-safety block; discovery
  treats it as reachable and keeps the model.
- `responses_400_access_denied.json` — an entitlement-error body
  (`access_denied`); no test loads it at present.

Chat completions API:

- `chat_completions_200_minimal.json` — a completed chat-completions response.
- `chat_completions_200_null_content.json` — a reasoning model's response with
  `content: null`, which the parser must treat as empty text.

Anthropic Messages API:

- `anthropic_messages_200_minimal.json` — a completed Messages response.
- `anthropic_messages_400_unsupported_temperature.json` — the message-only 400
  ("temperature is deprecated") that also triggers the temperature
  self-correction.

## Capture date

2026-06-19, for the Responses captures.

## Re-capture procedure (guarding against mock rot)

When AWS rolls a new Responses API revision, fixtures may diverge from
the live shape. To re-capture:

1. Set `AWS_PROFILE` to a profile with `bedrock-mantle:CreateInference`.
2. Run a real Responses call against the mantle endpoint (curl or the
   `BedrockMantleModel` adapter both work).
3. Replace the request and 200 fixtures with the new bodies (preserve
   the same model_id / input / max_output_tokens shape so the unit test
   contract stays).
4. Re-run `uv run pytest tests/test_bedrock_mantle_mappers.py
   tests/test_bedrock_mantle_model.py` — the schema-driven parser
   should still pass; if it doesn't, that is the canary.
5. Update this README's capture date.
