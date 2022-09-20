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

import * as fs from 'fs';
import * as Path from 'path';
import * as cudf from '@rapidsai/cudf';

const up = Path.dirname(__dirname).endsWith('lib') ? Path.join('..', '..') : '..';

type TaxiSchema = {
  Start_Lon: cudf.Float64,
  Start_Lat: cudf.Float64,
  End_Lon: cudf.Float64,
  End_Lat: cudf.Float64,
};

export function readPoints(
  base = Path.resolve(__dirname, up, 'data'),
  paths = fs.readdirSync(base)
    .filter((name) => name.endsWith('.parquet'))
    .slice(0, 1)
    .map((name) => Path.join(base, name))
) {

  const data = cudf.DataFrame.readParquet<TaxiSchema>(paths, {
    columns: ['Start_Lon', 'Start_Lat', 'End_Lon', 'End_Lat']
  });

  const x = data.select(['Start_Lon', 'End_Lon']).interleaveColumns();
  const y = data.select(['Start_Lat', 'End_Lat']).interleaveColumns();

  const yMask = y.ge(-90).logicalAnd(y.le(90));
  const xMask = x.ge(-180).logicalAnd(x.le(180));

  const points = new cudf.DataFrame({ x, y, }).filter(xMask.logicalAnd(yMask));

  const [xMin, xMax] = points.get('x').minmax();
  const [yMin, yMax] = points.get('y').minmax();

  return {
    bbox: [xMin, xMax, yMin, yMax] as [number, number, number, number],
    points
  };
}
