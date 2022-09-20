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

import * as maplibre from 'maplibre-gl';
import { MapboxOverlay } from '@deck.gl/mapbox/typed';
import {
  PathLayer,
  // GeoJsonLayer,
  // PolygonLayer,
  // ScatterplotLayer
} from '@deck.gl/layers/typed';

import { ColorMapper } from './color';
import { readStreets } from './read-streets';
import { readPoints } from './read-points-parquet';
import { makeQuadtree } from './make-tree';

// Read the 2019 Taxi pick-up/drop-off parquet files
// const { bbox, points } = readPoints();

// Spatially index the points in a Quadtree with cuSpatial
// const quadtree = makeQuadtree({ bbox, points });

// Read the NYC roadmap into GPU memory
const streets = readStreets();

console.log(streets
  .gather(cudf.Series.sequence({ init: 0, size: 1 }))
  .toArrow().toArray().map((x) => x.toJSON()));

// Assign a color to each street
const streetsWithColors = streets.assign({
  // Generate nice colors
  color: new ColorMapper().palette(0, streets.numRows, true)
});

// Flatten lists into line segments
const linesWithColors = streetsWithColors.flatten();

const lines = linesWithColors.get('geom');
const colors = linesWithColors.get('color');

// Center on the NYC road map centroid
const points = lines.flatten();
const x = points.getChild('x');
const y = points.getChild('y');
const center = [x.sum() / x.length, y.sum() / y.length] as [number, number];

// console.time(`copy roads DtoH (${streets.numRows.toLocaleString()} lines)`);

// Copy device -> host for deck.gl
const road_offsets = lines.offsets.toArray().subarray(0, -1);
const road_colors = colors.view(new cudf.Uint8).toArray();
const road_vertices = new cudf.DataFrame({ x, y }).interleaveColumns().toArray();

// console.timeEnd(`copy roads DtoH (${streets.numRows.toLocaleString()} lines)`);


// Add MapLibre GL for the basemap
const map = new maplibre.Map({
  interactive: true, container: document.body,
  zoom: 10, pitch: 0, bearing: 0, maxZoom: 20, center,
  style: 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json',
});

map.scrollZoom.setWheelZoomRate(1 / 25);

const deck = new MapboxOverlay({
  interleaved: true,
  layers: [
    new PathLayer({
      id: 'roads',
      opacity: 0.5,
      _pathType: 'loop',
      widthMinPixels: 2,
      positionFormat: `XY`,
      data: {
        length: streets.numRows,
        startIndices: road_offsets,
        attributes: {
          getPath: { value: road_vertices, size: 2 },
          getColor: { value: road_colors, size: 4, normalized: true }
        }
      },
    }),
  ]
});

map.addControl(deck as any);
