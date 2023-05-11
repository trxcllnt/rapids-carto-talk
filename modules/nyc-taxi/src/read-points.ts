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
import * as cudf from '@rapidsai/cudf';
import * as cuspatial from '@rapidsai/cuspatial';
import datasets from '@rapids-carto-talk/datasets';

export function readPoints(path = datasets.points) {

  type Point = cuspatial.Point<cudf.Float32>['TChildren'];

  const points = cudf.DataFrame.fromArrow<Point>(fs.readFileSync(path));

  console.log(`read ${points.numRows.toLocaleString()} points`);

  const [xMin, xMax] = points.get('x').minmax();
  const [yMin, yMax] = points.get('y').minmax();

  return {
    bbox: [xMin, xMax, yMin, yMax] as [number, number, number, number],
    points
  };
}
