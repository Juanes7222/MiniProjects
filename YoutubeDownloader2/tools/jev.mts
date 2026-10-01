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

async function main(): Promise<void> {
  const apiKey = process.env.AI_GATEWAY_API_KEY;
  if (!apiKey) {
    throw new Error('AI_GATEWAY_API_KEY is required for Jev');
  }

  const request = JSON.parse(await readStdin());
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
  process.stdout.write(JSON.stringify({ answers: result.answers }));
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : 'Unknown Jev bridge error';
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
}