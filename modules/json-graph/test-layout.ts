// Copyright (c) 2022, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import * as cudf from '@rapidsai/cudf';

import { readJSON } from './src/gpu-read';
import { shapeGraph } from './src/shape';
import { runLayout } from './src/layout';

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

(async () => {
  let i = 0;
  for await (const { edges, nodes, onAfterRender } of runLayout(shapeGraph(readJSON()))) {

    await new Promise<void>((r) => {
      process.stdout.cursorTo(0, 0, r);
    });

    await new Promise<void>((r) => {
      process.stdout.write(`\
(${i}) edges (length=${edges.length}):
${indent(new cudf.DataFrame(edges.attributes as any).toString({ maxRows: 5 }))}

(${i}) nodes (length=${nodes.length}):
${indent(new cudf.DataFrame(nodes.attributes as any).toString({ maxRows: 20 }))}
`, () => r());
    });

    ++i;

    onAfterRender();
  }
})()
  .catch((e) => { console.error(e); process.exit(1); });

function indent(str: string, chars = `  `) {
  return str.split('\n').map((line) => `${chars}${line}`).join(`\n`);
}
