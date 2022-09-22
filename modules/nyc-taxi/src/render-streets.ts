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
import { GeoJsonLayer } from '@deck.gl/layers/typed';
import { MapboxOverlay } from '@deck.gl/mapbox/typed';

import { ColorMapper } from './color';
import { readStreets } from './read-streets';

console.time('parse lines.csv');

// Flatten geometry lists into line segments
const streets = readStreets().select(['geom']);

console.timeEnd('parse lines.csv');

// Assign a color to each street
const colors = new ColorMapper();

// Center on the NYC road map centroid
const center = cudf.scope(() => {
  const points = streets.flatten().flatten().get('geom');
  const x = points.getChild('x');
  const y = points.getChild('y');
  return [
    x.sum() / x.length,
    y.sum() / y.length,
  ] as [number, number];
}, [streets]);

console.time(`copy streets DtoH (${streets.numRows.toLocaleString()} streets)`);

// Copy the multilines from device to host for deck.gl
const data = [...streets.toArrow()].map(({ geom }, i) => {
  const [r, g, b, a] = colors.get(i);
  const coordinates = [...geom].map((line) => {
    // points
    return [...line].map(({ x, y }) => [x, y]);
  });
  return {
    type: 'Feature',
    properties: {
      // Assign a color to each street
      color: [b, g, r, a],
    },
    geometry: {
      type: 'MultiLineString', coordinates,
    }
  };
});

console.timeEnd(`copy streets DtoH (${streets.numRows.toLocaleString()} streets)`);

// compare to geojson 
// import * as Path from 'path';
// const up = Path.dirname(__dirname).endsWith('lib') ? Path.join('..', '..') : '..';
// const data = JSON.parse(require('fs').readFileSync(Path.resolve(__dirname, up, 'data', 'NYC Street Centerline (CSCL).geojson')));

const map = new maplibre.Map({
  interactive: true, container: document.body,
  zoom: 10, pitch: 0, bearing: 0, maxZoom: 20, center,
  style: 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json',
});

map.scrollZoom.setWheelZoomRate(1 / 25);

const deck = new MapboxOverlay({
  interleaved: true,
  layers: [
    new GeoJsonLayer({
      id: 'geojson',
      data: data,
      filled: true,
      stroked: true,
      pickable: true,
      autoHighlight: true,
      lineWidthMinPixels: 3,
      getLineWidth: 1,
      getFillColor: [0, 0, 0, 0],
      getLineColor: [255, 255, 255, 135],
      highlightColor: (x) => x.object?.properties?.color
    }),
  ]
});

map.addControl(deck as any);
