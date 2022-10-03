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

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

import { Deck } from '@deck.gl/core/typed';
import { GraphLayer } from '@rapidsai/deck.gl';
import { PolygonLayer } from '@deck.gl/layers/typed';

import { ColorMapper } from './color';
import { readPolys } from './read-polys';
import { readPoints } from './read-points';
import { makeQuadtree } from './make-tree';
import { darkOrthoView, centerOnBbox } from './utils';

// Read the taxi zones dataset into GPU memory
const { bbox: polys_bbox, polys } = readPolys();

// Read the pick-up/drop-off locations into GPU memory
const { bbox: point_bbox, points } = readPoints();

// Spatially index the points in a Quadtree with cuSpatial
const quadtree = makeQuadtree({ bbox: point_bbox, points });

// Create a ColorMapper instance to generate nice colors
const colors = new ColorMapper();
const palette = colors.palette(0, polys.numRows, true);

// Copy the polygons from device to host for deck.gl
// console.time(`copy census tracts DtoH (${polys.numRows.toLocaleString()} tracts)`);
const polys_host = [...polys.toArrow()].map(({ id, polygon }) => {
  const rings = [...polygon].map((ring) => {
    // points
    return [...ring].map(({ x, y }) => [x, y]);
  });
  return { id, rings, color: colors.get(id) };
});
// console.timeEnd(`copy census tracts DtoH (${polys.numRows.toLocaleString()} tracts)`);

const deck = new Deck({
  views: [darkOrthoView],
  initialViewState: centerOnBbox(polys_bbox),
  controller: {
    keyboard: false,
    doubleClickZoom: false,
  },
  layers: [
    taxiZonesLayer(polys),
    taxiPointsLayer()
  ]
});

function taxiZonesLayer(
  polys: ReturnType<typeof readPolys>['polys'],
  hoveredPolygonIndex = -1,
) {
  return new PolygonLayer({
    id: 'polys',
    data: polys_host,
    opacity: 0.5,
    filled: true,
    stroked: true,
    pickable: true,
    extruded: false,
    positionFormat: `XY`,
    lineWidthMinPixels: 1,
    getElevation: ({ id }) => id,
    getPolygon: ({ rings }) => rings,
    getFillColor: ({ color: [r, g, b] }: any) => [r, g, b, 15] as any,
    getLineColor: ({ color: [r, g, b] }: any) => [r, g, b, 255] as any,
    onHover: ({ index, picked, object }) => {
      if (!picked || index === -1) {
        deck.setProps({
          layers: [
            taxiZonesLayer(polys, hoveredPolygonIndex = -1),
            taxiPointsLayer()
          ]
        });
      } else if (
        (object && typeof object.id === 'number') &&
        (object.id !== hoveredPolygonIndex)
      ) {
        const { hoveredPolyId, polyPointPairs } = cudf.scope(() => {

          // Gather a dataframe of only the hovered polygon
          const hoveredPolyId = cudf.Series.sequence({ init: object.id, size: 1 });
          const hoveredPoly = polys.gather(hoveredPolyId).get('polygon');

          console.time('point in polygon runtime');

          // Spatial join for points intersect w/ the hovered polygon
          const polyPointPairs = quadtree.pointInPolygon(hoveredPoly);

          console.timeEnd('point in polygon runtime');
          console.log(`points in poly ${object.id}:`, polyPointPairs.numRows.toLocaleString());

          return { hoveredPolyId, polyPointPairs };
        }, [polys, quadtree, palette]);

        deck.setProps({
          layers: [
            taxiZonesLayer(polys, hoveredPolygonIndex = object.id),
            taxiPointsLayer(
              hoveredPolyId.gather(polyPointPairs.get('polygon_index')),
              polyPointPairs.get('point_index'),
            )
          ]
        });
      }
      return true;
    },
  });
}

function taxiPointsLayer(
  polyIds = cudf.Series.sequence({ init: 0, size: 0, type: new cudf.Int32 }),
  pointIds = cudf.Series.sequence({ init: 0, size: 0, type: new cudf.Uint32 })
) {
  const length = pointIds.length;
  const props = { ...defaultPointProps(), numNodes: length };

  if (length > 0) {
    // Gather colors using the poly ids from spatial join results
    const colors = palette.gather(polyIds);
    // Gather points using the point ids from spatial join results
    const points = quadtree.points.gather(pointIds);

    Object.assign(props.data.nodes, {
      length,
      attributes: {
        nodeFillColors: colors.data,
        nodeElementIndices: polyIds.data,
        nodeXPositions: points.get('x').data,
        nodeYPositions: points.get('y').data,
        nodeRadius: cudf.Series.sequence({
          init: 5,
          step: 0,
          size: length,
          type: new cudf.Uint8,
        }).data,
      }
    });
  }

  return new GraphLayer(props) as any;
}

import { log as deckLog } from '@deck.gl/core/typed';
deckLog.level = 0;
deckLog.enable(false);

function defaultPointProps() {
  return {

    nodesVisible: true,
    nodesFilled: true,
    nodesStroked: false,
    nodeRadiusScale: 10,
    nodeRadiusMinPixels: 0,
    nodeRadiusMaxPixels: 1,

    numEdges: 0,
    edgesVisible: false,

    _subLayerProps: {
      'NodeLayer-0': { pickable: false, autoHighlight: false, },
      'EdgeLayer-0': { pickable: false, autoHighlight: false, },
    },
    data: {
      edges: {
        length: 0,
        offset: 0,
        attributes: {}
      },
      nodes: {
        length: 0,
        offset: 0,
        attributes: {}
      }
    },

  };
}
