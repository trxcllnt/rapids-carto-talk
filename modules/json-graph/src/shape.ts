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
import * as cugraph from '@rapidsai/cugraph';

type JSONGraph = ReturnType<typeof import('./gpu-read').readJSON>;
type InNodes = JSONGraph['nodes']['types'];

export type ShapedNodes = ReturnType<typeof shapeNodes>;
export type ShapedEdges = ReturnType<typeof shapeEdges>;

export function shapeGraph({ edges: inEdges, nodes: inNodes }: JSONGraph) {

  inEdges = inEdges.gather(cudf.Series.sequence({
    init: 0,
    step: 2,
    size: Math.min(1e5 * 2, inEdges.numRows / 2)
  }));

  const graph = cugraph.Graph.fromEdgeList(
    inEdges.get('source'), inEdges.get('target')
  );

  const nodes = shapeNodes(graph, inNodes);
  const edges = shapeEdges(graph, nodes);

  return { nodes, edges, graph };
}

function shapeNodes(
  graph: cugraph.Graph<cudf.Int32>,
  inNodes: cudf.DataFrame<InNodes>,
) {

  const nodes = graph.nodes.assign(graph.nodeIds).join({
    on: ['node'],
    other: inNodes.rename({ key: 'node' })
  }).drop(['node']);

  // Scale and cast size to Uint8 for shading
  const size = nodes.get('size').scale().mul(254).add(1).cast(new cudf.Uint8);
  // Cast the CSS hex colors to Int32 for shading
  // Replace leading # with FF before casting hex string -> int
  const color = nodes.get('color').replaceSlice('FF', 0, 1).hexToIntegers(new cudf.Int32);

  return nodes
    .assign({ size, color })
    .sortValues({ id: { ascending: true } })
    .select(['id', 'x', 'y', 'size', 'color']);
}

function shapeEdges(graph: cugraph.Graph<cudf.Int32>, nodes: ShapedNodes) {
  const edges = computeEdgeBundles(graph.edgeIds);
  const color = computeEdgeColors(edges, nodes);
  return edges.assign({ color });
}

function scoped<F extends (...args: any[]) => any>(fn: F) {
  return (...args: Parameters<F>) => {
    return cudf.scope(() => fn(...args), args) as ReturnType<F>;
  }
}

const computeEdgeColors = scoped((edges: ReturnType<typeof computeEdgeBundles>, nodes: ShapedNodes) => {

  const src = edges.select(['id', 'src'])
    .join({
      on: ['src'],
      other: nodes.select(['id', 'color']).rename({ id: 'src' })
    })
    .sortValues({ id: { ascending: true } })
    .get('color');

  const dst = edges.select(['id', 'dst'])
    .join({
      on: ['dst'],
      other: nodes.select(['id', 'color']).rename({ id: 'dst' })
    })
    .sortValues({ id: { ascending: true } })
    .get('color');

  // Interleave each Int32 src/dst color into a single stride-2 Uint64
  return new cudf.DataFrame({ src, dst }).interleaveColumns().view(new cudf.Uint64);
});

const edgesToBundles = scoped((edges: cudf.DataFrame<{
  id: cudf.Int32, src: cudf.Int32, dst: cudf.Int32,
}>) => {

  const groups = edges.groupBy({
    by: ['src', 'dst'], index_key: 'src_dst',
  });

  // Compute the number of edges in each bundle
  const sizes = groups.count().rename({ id: 'length' });

  // Collect each bundle into sub-lists of edge ids
  const lists = groups.collectList().rename({ id: 'offset' });

  return sizes
    // Join the edge bundle counts and edge id lists into one dataframe for flattening
    .join({ on: ['src_dst'], other: lists })
    // Flatten the dataframe's edge bundle sub-lists and take the index
    // of each element in each sub-list (instead of each element value)
    .flattenIndices(['offset']);
});

const computeEdgeBundles = scoped((edges: cudf.DataFrame<{ id: cudf.Int32, src: cudf.Int32, dst: cudf.Int32 }>) => {

  const bundles = edgesToBundles(edges);

  const edgesAndBundles = edges.join({
    // Join on the `src` and `dst` columns:
    on: ['src', 'dst'],

    // other: bundles,

    // Extract the src/dst columns for joining with the original edges
    other: bundles.drop(['src_dst']).assign({
      src: bundles.get('src_dst').getChild('src'),
      dst: bundles.get('src_dst').getChild('dst'),
    })
  });

  return edgesAndBundles
    .assign({
      // interleave the Int32 edge src/dst columns into a single stride-2 Uint64 column
      edge: edgesAndBundles.select(['src', 'dst']).interleaveColumns().view(new cudf.Uint64),
      // interleave the Int32 bundle offset and length into a single stride-2 Uint64 column
      bundle: edgesAndBundles.select(['offset', 'length']).interleaveColumns().view(new cudf.Uint64),
    })
    .select(['id', 'src', 'dst', 'edge', 'bundle'])
    .sortValues({ id: { ascending: true } });
});
