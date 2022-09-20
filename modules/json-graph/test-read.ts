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

import * as util from 'util';
import * as rmm from '@rapidsai/rmm';
import * as cudf from '@rapidsai/cudf';
import { readJSON as readJSONCPU } from './src/cpu-read';
import { readJSON as readJSONGPU } from './src/gpu-read';

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

parse_json_print_results('CPU', readJSONCPU);

parse_json_print_results('GPU', readJSONGPU);

initPoolAllocator();
parse_json_print_results('RMM', readJSONGPU);


function parse_json_print_results(suf: string, readJSON: typeof readJSONCPU | typeof readJSONGPU) {
  cudf.scope(() => {
    console.log(`\n#############################################################`);

    console.time(`(${suf}) read + parse time`);
    let { options, nodes, edges } = readJSON(false);
    console.timeEnd(`(${suf}) read + parse time`);

    if (!(nodes instanceof cudf.DataFrame)) {
      console.time(`(${suf}) copy nodes to GPU`);
      nodes = new cudf.DataFrame(nodes as any);
      console.timeEnd(`(${suf}) copy nodes to GPU`);
    }

    if (!(edges instanceof cudf.DataFrame)) {
      console.time(`(${suf}) copy edges to GPU`);
      edges = new cudf.DataFrame(edges as any);
      console.timeEnd(`(${suf}) copy edges to GPU`);
    }

    console.log(`
(${suf}) options: ${util.inspect(options)}`);

    console.log(`
(${suf}) nodes (len=${nodes.numRows}):
${indent(nodes.toString({ maxRows: 8 }))}`);

    console.log(`
(${suf}) edges (len=${edges.numRows}):
${indent(edges.toString({ maxRows: 8 }))}`);

    console.log(`#############################################################\n`);
  });
}

function indent(str: string, chars = `  `) {
  return str.split('\n').map((line) => `${chars}${line}`).join('\n');
}

function initPoolAllocator() {
  rmm.setCurrentDeviceResource(new rmm.PoolMemoryResource(
    rmm.getCurrentDeviceResource(),
    // Number(1n << 32n), // 8 GB
    // Number(1n << 34n) // 16 GB
  ));
}
