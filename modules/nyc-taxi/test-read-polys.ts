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

import { readPolys } from './src/read-polys';

console.time('read polys runtime');

const polys = readPolys();

console.timeEnd('read polys runtime');

console.log(require('util').inspect(polys));
