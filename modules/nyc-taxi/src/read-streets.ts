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
import datasets from '@rapids-carto-talk/datasets';

type NYCCenterlineSchema = { the_geom: cudf.Utf8String, FULL_STREE: cudf.Utf8String };

export function readStreets(path = datasets.centerline) {
  return cudf.scope(() => {

    // Read the geometry column from the CSV
    const streetsAndGeom = cudf.DataFrame
      .readCSV<NYCCenterlineSchema>(path)
      .select(['the_geom', 'FULL_STREE'])
      .rename({ FULL_STREE: 'name', the_geom: 'geom' });

    console.log(streetsAndGeom.numRows.toLocaleString());

    // Filter for lines that start with MULTILINESTRING
    const multiLineStreets = streetsAndGeom
      // .assign({ name: streetsAndGeom.get('name').encodeLabels() })
      .filter(streetsAndGeom.get('geom').matchesRe(/MULTILINESTRING\s?/));

    // Parse each geometry string into linestrings (lists of X/Y points)
    const geom = parseLineStrings(multiLineStreets.get('geom'));

    // Group each line segment by the street it belongs to to construct the multi-line string for each street
    return multiLineStreets.assign({ geom }).groupBy({ by: 'name' }).collectList();
  });
}

function parseLineStrings(geom: cudf.Series<cudf.Utf8String>) {

  geom = geom
    // Remove 'MULTILINESTRING ' prefix
    .replaceRe(/MULTILINESTRING\s?/, '')
    // Remove open parentheses
    .replaceRe(/\(/, '')
    // Replace double close parentheses with | separator
    .replaceRe(/\)\)/, '|')
    // Replace pair delimiter with | separator
    .replaceRe(/,\s/, '|');

  // Compute keys used to group the points into linestring lists
  const id = cudf.scope(() => {
    const sums = geom
      // Count the number of pairs in each row to compute scatter map
      .countRe(/\|/)
      // Cumulative sum to compute the scatter map indices
      .cumulativeSum().cast(new cudf.Int32);

    // Start with a list of zeroes
    return cudf.Series.sequence({ init: 0, step: 0, size: sums.max() + 1 })
      // Scatter 1's into place using the cumulative sums above
      .scatter(1, sums, true)
      // Cumulative sum to produce the id (i.e. exclusive_scan)
      .cumulativeSum();
  }, [geom]);

  // Flatten and extract the LON/LAT values into separate columns
  const points = cudf.scope(() => {
    const [x, , y] = geom
      // Flatten each linestring into point pairs
      .split('|')
      // Remove the trailing | from each pair
      .replaceRe(/\|/, '')
      // Extract the `x y` pairs into separate columns
      .partition(' ');

    return new cudf.DataFrame({ x, y })
      // Cast x/y strings -> Float32
      .castAll(new cudf.Float32)
      // Return the DataFrame as a Struct column
      .asStruct();
  }, [geom]);

  // Group the flattened points into lines by id keys
  const lines = new cudf.DataFrame({ id, points })
    // The last element was the extra slot for the final sum
    // in the exclusive_scan, remove it now
    .head(id.length - 1)
    // Group the x/y columns by the exclusive_scan id
    .groupBy({ by: 'id' })
    .collectList()
    .sortValues({ id: { ascending: true } })
    .get('points');

  return lines;
}
