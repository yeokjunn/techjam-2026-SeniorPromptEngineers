#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { validateSlides } from '../lib/validate.js';

const files = process.argv.slice(2);

if (files.length === 0) {
  console.error('Usage: validate-slides <file.html> [file2.html ...]');
  process.exit(1);
}

let hasFailure = false;

for (const file of files) {
  const filePath = path.resolve(file);

  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    hasFailure = true;
    continue;
  }

  const html = fs.readFileSync(filePath, 'utf-8');
  const results = validateSlides(html);

  console.log(`\n  ${path.basename(filePath)}`);
  console.log('  ' + '-'.repeat(40));

  for (const r of results) {
    const icon = r.passed ? '\x1b[32m✓\x1b[0m' : '\x1b[31m✗\x1b[0m';
    console.log(`  ${icon} ${r.rule}: ${r.message}`);
  }

  const passed = results.filter(r => r.passed).length;
  const total = results.length;
  console.log(`\n  ${passed}/${total} rules passed\n`);

  if (passed < total) hasFailure = true;
}

process.exit(hasFailure ? 1 : 0);
