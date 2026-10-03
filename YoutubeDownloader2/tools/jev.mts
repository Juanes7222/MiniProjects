import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadEnvFile } from 'node:process';

// Transparent bridge: Python owns the question contract (what is asked, the
// criteria, how answers combine) and hands us a ready System One request. This
// process only authenticates and forwards, so a question change never has to be
// mirrored here.
const envPath = fileURLToPath(new URL('../.env.local', import.meta.url));
if (existsSync(envPath)) {
  loadEnvFile(envPath);
}

async function readStdin(): Promise<string> {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
  }
  return input;
}

async function evaluateRequest(request: any): Promise<Record<string, unknown>> {
  const apiKey = process.env.AI_GATEWAY_API_KEY;
  if (!apiKey) {
    throw new Error('AI_GATEWAY_API_KEY is required for Jev');
  }

  const state = request.state;
  const questions = request.questions;
  if (state === undefined || questions === undefined) {
    throw new Error('Request must contain `state` and `questions`');
  }

  const response = await fetch('https://ai-gateway.vercel.sh/v4/ai/evaluation-model', {
    method: 'POST',
    headers: {
      authorization: `Bearer ${apiKey}`,
      'ai-evaluation-model-specification-version': '4',
      'ai-gateway-auth-method': 'api-key',
      'ai-gateway-protocol-version': '0.0.1',
      'ai-model-id': request.model ?? 'typesafe-ai/jev',
      'content-type': 'application/json',
    },
    body: JSON.stringify({ state, model: request.model, questions }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`AI Gateway returned ${response.status}: ${detail}`);
  }

  const result = await response.json();
  return { answers: result.answers };
}

// One-shot: read a single request from stdin, write one response to stdout,
// then exit. Kept for callers that want a fresh process per evaluation.
async function main(): Promise<void> {
  const request = JSON.parse(await readStdin());
  process.stdout.write(JSON.stringify(await evaluateRequest(request)));
}

// Long-lived mode: newline-delimited JSON in, newline-delimited JSON out.
//
// Spawning Node and re-loading the runtime for every song is a fixed cost paid
// per candidate set, on top of a network round trip that is far larger. Keeping
// one process for the whole run removes that cost entirely. Requests are still
// answered strictly in order, one line in to one line out, which is what lets
// the Python side keep its retries and error handling unchanged.
//
// A failing request answers with {"error": ...} and the loop continues: one bad
// evaluation must not take the bridge -- and every evaluation after it -- down.
async function serve(): Promise<void> {
  let buffer = '';
  process.stdin.setEncoding('utf8');

  for await (const chunk of process.stdin) {
    buffer += chunk;

    let newline = buffer.indexOf('\n');
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);

      if (line.length > 0) {
        let response: Record<string, unknown>;
        try {
          response = await evaluateRequest(JSON.parse(line));
        } catch (error) {
          response = {
            error: error instanceof Error ? error.message : 'Unknown Jev bridge error',
          };
        }
        process.stdout.write(`${JSON.stringify(response)}\n`);
      }

      newline = buffer.indexOf('\n');
    }
  }
}

try {
  if (process.argv.includes('--server')) {
    await serve();
  } else {
    await main();
  }
} catch (error) {
  const message = error instanceof Error ? error.message : 'Unknown Jev bridge error';
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
}