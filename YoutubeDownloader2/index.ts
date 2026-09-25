import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadEnvFile } from 'node:process';
import { generateText } from 'ai';

const envPath = fileURLToPath(new URL('./.env.local', import.meta.url));
if (existsSync(envPath)) {
  loadEnvFile(envPath);
}

const result = await generateText({
  model: 'openai/gpt-5.5',
  prompt: 'Invent a new holiday and describe its traditions.',
});

console.log(result.text);
