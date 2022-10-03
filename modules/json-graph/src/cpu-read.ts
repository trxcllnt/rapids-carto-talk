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

import * as fs from 'fs';
import * as Path from 'path';
import datasets from '@rapids-carto-talk/datasets';

export function readJSON(quiet = true, path = datasets.graph) {

  const name = Path.basename(path);

  !quiet && console.time(`(CPU) read '${name}'`);

  const file = fs.readFileSync(path, 'utf8');

  !quiet && console.timeEnd(`(CPU) read '${name}'`);

  !quiet && console.time(`(CPU) JSON.parse '${name}'`);

  const { nodes, edges, ...json } = JSON.parse(file);

  !quiet && console.timeEnd(`(CPU) JSON.parse '${name}'`);

  !quiet && console.time(`(CPU) create node list`);

  const nodeKey = new Int32Array(nodes.length);
  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeSize = new Int32Array(nodes.length);
  const nodeColor = new Array(nodes.length);
  for (let i = -1, n = nodes.length; ++i < n;) {
    const { key, attributes } = nodes[i];
    nodeKey[i] = +key;
    nodeX[i] = +attributes.x;
    nodeY[i] = +attributes.y;
    nodeSize[i] = +attributes.size;
    nodeColor[i] = attributes.color;
  }

  !quiet && console.timeEnd(`(CPU) create node list`);

  !quiet && console.time(`(CPU) create edge list`);

  const edgeKey = new Array(edges.length);
  const edgeSource = new Int32Array(edges.length);
  const edgeTarget = new Int32Array(edges.length);
  for (let i = -1, n = edges.length; ++i < n;) {
    const { key, source, target } = edges[i];
    edgeKey[i] = key;
    edgeSource[i] = +source;
    edgeTarget[i] = +target;
  }

  !quiet && console.timeEnd(`(CPU) create edge list`);

  return {
    options: json.options,
    nodes: {
      key: nodeKey,
      x: nodeX,
      y: nodeY,
      size: nodeSize,
      color: nodeColor
    },
    edges: {
      key: edgeKey,
      source: edgeSource,
      target: edgeTarget,
    }
  };
}
