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

import * as cuda from '@rapidsai/cuda';
import * as cudf from '@rapidsai/cudf';

type ShapedResult = ReturnType<typeof import('./shape').shapeGraph>;

export async function* runLayout(props: ShapedResult) {

  let n = props.graph.numNodes;
  let onAfterRender = () => { };
  let afterRender = Promise.resolve();
  let positions: cuda.Float32Buffer | undefined;

  // Use the initial X and Y positions to start
  positions = props.nodes.get('x')
  /**/.concat(props.nodes.get('y'))
  /**/.data;

  while (true) {

    if (positions) {
      // Make a promise for deck.gl to fulfill after it renders a frame
      afterRender = new Promise((r) => onAfterRender = r);

      // Yield the graph attrs to the render loop
      yield attrs(props, onAfterRender);

      // Wait for deck.gl to render the frame
      await afterRender;
    }

    do {
      // Run layout to compute new positions from current positions
      positions = props.graph.forceAtlas2({ positions, ...params });

      // forceAtlas2's integration occasionally encounters degenerate
      // cases and produces a few NaN positions. This will cause the
      // next layout tick's integration to produce nearly entirely
      // NaN positions.
      //
      // Detect this condition and allow cuGraph to re-initialize the
      // positions randomly, as they'll eventually re-converge to the
      // same positions over time.
      if (cudf.Series.new(positions).isNaN().sum() > 0) {
        positions = undefined;
        continue;
      }
      break;
    } while (true);

    // Extract X/Ys and reassign them to the nodes df
    props.nodes = props.nodes.assign({
      x: cudf.Series.new(positions.subarray(0, n * 1)),
      y: cudf.Series.new(positions.subarray(n, n * 2)),
    });
  }
}

const params = {
  gravity             /**/: 2.0,
  scalingRatio        /**/: 1.0,
  linLogMode          /**/: !true,
  strongGravityMode   /**/: true,
  outboundAttraction  /**/: !true,
  edgeWeightInfluence /**/: 0.0,
  barnesHutTheta      /**/: 1e-9,
  jitterTolerance     /**/: 0.05,
};

const attrs = (
  { graph, nodes, edges }: ShapedResult,
  onAfterRender: () => any
) => ({
  // Hook our layout loop into deck.gl's render loop
  onAfterRender,
  // Compute the positions bounding box [xMin, xMax, yMin, yMax]
  bbox: [
    ...nodes.get('x').minmax(),
    ...nodes.get('y').minmax()
  ],
  // Deck node and edge layer attributes
  nodes: {
    offset: 0,
    length: graph.numNodes,
    attributes: {
      nodeRadius: nodes.get('size').data,
      nodeXPositions: nodes.get('x').data,
      nodeYPositions: nodes.get('y').data,
      nodeFillColors: nodes.get('color').data,
      nodeElementIndices: nodes.get('id').data,
    }
  },
  edges: {
    offset: 0,
    length: graph.numEdges,
    attributes: {
      edgeList: edges.get('edge').data,
      edgeColors: edges.get('color').data,
      edgeBundles: edges.get('bundle').data,
    }
  },
});
