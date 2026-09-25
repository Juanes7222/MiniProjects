import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadEnvFile } from 'node:process';
import { experimental_evaluate } from 'ai';

const envPath = fileURLToPath(new URL('../.env.local', import.meta.url));
if (existsSync(envPath)) {
  loadEnvFile(envPath);
}

let input = '';
for await (const chunk of process.stdin) {
  input += chunk;
}

const payload = JSON.parse(input);
const questions = Object.fromEntries(
  payload.candidates.map((candidate: { key: string; label: string }) => [
    candidate.key,
    {
      type: 'boolean',
      instructions: `Does ${candidate.label} represent exactly the requested song by the requested artist? Return true only for the original recording or the official track. Return false for a cover, live version, remix, instrumental, karaoke, lyric-only upload, reaction, compilation, or unrelated song.`,
    },
  ]),
);

const result = await experimental_evaluate({
  model: 'typesafe-ai/jev',
  state: payload.state,
  questions,
});

process.stdout.write(JSON.stringify({ answers: result.answers }));
