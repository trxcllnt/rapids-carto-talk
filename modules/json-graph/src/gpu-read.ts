// Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import * as cudf from '@rapidsai/cudf';
import datasets from '@rapids-carto-talk/datasets';

const _0 = cudf.Series.new(new Int32Array([0]));
const _1 = cudf.Series.new(new Int32Array([1]));

export function readJSON(quiet = true, path = datasets.graph) {

  const name = Path.basename(path);

  !quiet && console.time(`(GPU) read '${name}'`);

  const file = cudf.Series.new([fs.readFileSync(path, 'utf-8')]);

  !quiet && console.timeEnd(`(GPU) read '${name}'`);

  let opts: Options;
  let pair: cudf.Series<cudf.Utf8String>;
  let next: cudf.Series<cudf.Utf8String>;
  let edge: cudf.DataFrame<typeof EdgeTypes>;
  let node: cudf.DataFrame<typeof NodeTypes>;

  !quiet && console.time(`(GPU) read options`);

  pair = file.split('"options":');
  ({ next, opts } = readOptions(pair));

  !quiet && console.timeEnd(`(GPU) read options`);

  !quiet && console.time(`(GPU) create edge list`);

  pair = next.split('"edges":');
  ({ next, edge } = readEdges(pair));

  !quiet && console.timeEnd(`(GPU) create edge list`);

  !quiet && console.time(`(GPU) create node list`);

  pair = next.split('"nodes":');
  ({ next, node } = readNodes(pair));

  !quiet && console.timeEnd(`(GPU) create node list`);

  return { options: opts, nodes: node, edges: edge };
}

interface Options { type: string, multi: boolean, allowSelfLoops: boolean }

function readOptions(pair: cudf.Series<cudf.Utf8String>) {
  return {
    next: pair.gather(_0),
    opts: JSON.parse(
      pair.gather(_1)
        .split('}').gather(_0)
        .getValue(0)!
    ) as Options
  }
}

const EdgeTypes = {
  key: new cudf.Utf8String,
  source: new cudf.Int32,
  target: new cudf.Int32,
};

function readEdges(pair: cudf.Series<cudf.Utf8String>) {
  type E = typeof EdgeTypes;
  type K = string & keyof E;

  const types = Object.entries(EdgeTypes) as [K, E[K]][];
  const edges = pair.gather(_1)
    .split('[\n').gather(_1)
    .split('\n]').gather(_0)
    .split('},');

  const cols = {} as any;

  types.forEach(([name, dtype]) => {
    cols[name] = edges.getJSONObject(`.${name}`).cast(dtype);
  });

  return {
    next: pair.gather(_0),
    edge: new cudf.DataFrame<E>(cols).dropNulls(0, types.length),
  };
}

const NodeTypes = {
  key: new cudf.Int32,
  x: new cudf.Float32,
  y: new cudf.Float32,
  size: new cudf.Int32,
  color: new cudf.Utf8String,
};

function readNodes(pair: cudf.Series<cudf.Utf8String>) {
  type N = typeof NodeTypes;
  type K = string & keyof N;

  const types = Object.entries(NodeTypes) as [K, N[K]][];
  const nodes = pair.gather(_1)
    .split('[\n').gather(_1)
    .split('\n]').gather(_0)
    .split('},');

  const cols = {} as any;

  types.forEach(([name, dtype]) => {
    const attr = name === 'key' ? name : `attributes.${name}`;
    cols[name] = nodes.getJSONObject(`.${attr}`).cast(dtype);
  });

  return {
    next: pair.gather(_0),
    node: new cudf.DataFrame<N>({
      ...cols,
      x: cols.x.replaceNulls(0),
      y: cols.y.replaceNulls(0),
      size: cols.size.replaceNulls(0),
      color: cols.color.replaceNulls('#000000'),
    }),
  };
}
