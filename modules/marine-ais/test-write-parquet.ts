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

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

require('rimraf').sync('./data/*.parquet');

const files =
  fs.readdirSync('./data')
    .filter((f) => f.endsWith('.zip'))
    .slice(0, 2)
    .map((name) => `data/${name}`);

files.forEach((file) => {

  const { name } = Path.parse(file);

  console.time(`read "${file}"`);

  const data = cudf.DataFrame
    .readCSV(file, { compression: 'infer' })
    .cast({
      BaseDateTime: new cudf.TimestampSecond
    });

  console.timeEnd(`read "${file}"`);


  console.time(`write ${name}.parquet`);

  data.toParquet(`./data/${name}.parquet`, { compression: 'snappy' });

  console.timeEnd(`write ${name}.parquet`);
});
