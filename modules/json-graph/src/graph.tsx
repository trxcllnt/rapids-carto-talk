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

import * as React from 'react';
import DeckGL from '@deck.gl/react/typed';
import { GraphLayer } from '@rapidsai/deck.gl';
import { OrthographicView } from '@deck.gl/core/typed';

import { readJSON } from './gpu-read';
import { shapeGraph } from './shape';
import { runLayout } from './layout';

export default () => {

  const [bbox, setBoundingBox] = React.useState(bbox_nans);
  const [autoCenter, setAutoCenter] = React.useState(true);
  const [{ onAfterRender }, setOnAfterRender] =
    React.useState({ onAfterRender: undefined });

  const json = React.useMemo(() => readJSON(), []);
  const shaped = React.useMemo(() => shapeGraph(json), [json]);
  const updates = React.useMemo(() => runLayout(shaped), [shaped]);

  return (
    <DeckGL
      views={[ortho]}
      controller={true}
      onAfterRender={onAfterRender}
      onViewStateChange={() => setAutoCenter(false)}
      initialViewState={autoCenter && centerOnBbox(bbox)}
    >
      {/*@ts-expect-error*/}
      <GraphLayer
        data={updates}
        dataTransform={
          ({ bbox, onAfterRender, ...data }: any) => {
            setBoundingBox(bbox);
            setOnAfterRender({ onAfterRender });
            return data;
          }
        }
        edgeOpacity={.01}
        edgeStrokeWidth={2}
        nodesStroked={true}
        nodeFillOpacity={.0}
        nodeStrokeOpacity={.5}
        nodeRadiusScale={1 / 75}
        nodeRadiusMinPixels={5}
        nodeRadiusMaxPixels={50}
        numNodes={shaped.nodes.numRows}
        numEdges={shaped.edges.numRows}
      />
    </DeckGL>
  );
};

const bbox_nans = [NaN, NaN, NaN, NaN] as [number, number, number, number];

const ortho = new OrthographicView({
  clear: { color: [...[46, 46, 46].map((x) => x / 255), 1] }
} as any);

function centerOnBbox(
  [minX, maxX, minY, maxY]: [number, number, number, number],
  [parentWidth, parentHeight]: [number, number] = [window.outerWidth, window.outerHeight]
) {
  const width = Math.max(maxX - minX, 1);
  const height = Math.max(maxY - minY, 1);
  if ((width === width) && (height === height)) {
    const xRatio = width / parentWidth;
    const yRatio = height / parentHeight;
    let zoom: number;
    if (xRatio > yRatio) {
      zoom = ((width > parentWidth) ? -(width / parentWidth) : (parentWidth / width)) * .9;
    } else {
      zoom = ((height > parentHeight) ? -(height / parentHeight) : (parentHeight / height)) * .9;
    }
    return {
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
      zoom: Math.log2(Math.abs(zoom)) * Math.sign(zoom),
      target: [minX + (width * .5), minY + (height * .5), 0],
    };
  }
  return {
    zoom: 1,
    target: [0, 0, 0],
    minZoom: Number.NEGATIVE_INFINITY,
    maxZoom: Number.POSITIVE_INFINITY,
  };
}

import { log as deckLog } from '@deck.gl/core/typed';
deckLog.level = 0;
deckLog.enable(false);
