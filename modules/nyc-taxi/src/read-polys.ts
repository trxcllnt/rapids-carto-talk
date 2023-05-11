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
import * as arrow from 'apache-arrow';
import * as cudf from '@rapidsai/cudf';
import * as cuspatial from '@rapidsai/cuspatial';
import datasets from '@rapids-carto-talk/datasets';

export function readPolys(path = datasets.polys) {

  console.log('readPolys path: ' + path);

  const table = arrow.tableFromIPC<{
    index: cudf.Int32,
    tract: cuspatial.Polygon<cudf.Float32>
  }>(fs.readFileSync(path));

  const polys = new cudf.DataFrame({
    id: table.getChild('index')!,
    polygon: table.getChild('tract')!,
  });

  const points = polys.get('polygon').elements.elements;
  const [xMin, xMax] = points.getChild('x').minmax();
  const [yMin, yMax] = points.getChild('y').minmax();

  return {
    bbox: [xMin, xMax, yMin, yMax] as [number, number, number, number],
    polys,
  };
}
