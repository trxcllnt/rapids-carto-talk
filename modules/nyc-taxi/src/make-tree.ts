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

import * as cuspatial from '@rapidsai/cuspatial';

type BboxAndPoints = ReturnType<typeof import('./read-points').readPoints>;

export function makeQuadtree({ bbox, points }: BboxAndPoints) {

  const [xMin, xMax, yMin, yMax] = bbox;

  return cuspatial.Quadtree.new({
    x: points.get('x'),
    y: points.get('y'),
    xMin,
    xMax,
    yMin,
    yMax,
    scale: 0,
    maxDepth: 15,
    minSize: 1e5,
  });
}
