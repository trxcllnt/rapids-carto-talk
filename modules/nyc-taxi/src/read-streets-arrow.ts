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
import * as arrow from 'apache-arrow';
import * as cudf from '@rapidsai/cudf';

const up = Path.dirname(__dirname).endsWith('lib') ? Path.join('..', '..') : '..';

type NYCCenterlineSchema = {
  name: cudf.Utf8String,
  multiline: cudf.List<cudf.List<cudf.Struct<{
    x: cudf.Float64,
    y: cudf.Float64
  }>>>
};

export function readStreets(path = Path.resolve(__dirname, up, 'data', 'centerline.arrow')) {

  const table = arrow.tableFromIPC<NYCCenterlineSchema>(fs.readFileSync(path));

  return new cudf.DataFrame({
    name: table.getChild('name')!,
    multiline: table.getChild('multiline')!
  });
}
