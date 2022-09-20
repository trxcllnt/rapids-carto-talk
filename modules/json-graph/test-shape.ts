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

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

const { nodes, edges } = shapeGraph(readJSON());

console.log(`edges (len=${edges.numRows}):
${indent(edges.toString({ maxRows: 10 }))}`);

console.log(`nodes (len=${nodes.numRows}):
${indent(nodes.toString({ maxRows: 10 }))}`);

function indent(str: string, chars = `  `) {
  return str.split('\n').map((line) => `${chars}${line}`).join('\n');
}
