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
import * as util from 'util';
import * as cudf from '@rapidsai/cudf';

// Initialize GPU before measuring timings
cudf.Series.new([1, 2, 3]).sum();

const files = fs.readdirSync('./data')
  .filter((f) => f.endsWith('.parquet'))
  .slice(0, 2)
  .map((name) => `data/${name}`);

console.time(`read parquet file(s): [\n${indent(`"${files.join('",\n"')}"`)}\n]\n`);

let data = cudf.DataFrame.readParquet(files);

console.timeEnd(`read parquet file(s): [\n${indent(`"${files.join('",\n"')}"`)}\n]\n`);

console.log(`data types:
${indent(util.inspect(data.types))}\n`);

console.log(`data (len=${data.numRows.toLocaleString()}):
${indent(data.toString({ maxRows: 20 }))}\n`);

function indent(str: string, chars = `  `) {
  return str.split('\n').map((line) => `${chars}${line}`).join('\n');
}
