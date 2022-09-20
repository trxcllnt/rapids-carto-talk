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

import { Series, Uint32 } from '@rapidsai/cudf';

const goldenRatioConjugate = 0.618033988749895;

export class ColorMapper {
  private declare _h: number;
  private declare _s: number;
  private declare _v: number;
  private declare _map: any;
  constructor(hue = 0.99, saturation = 0.99, brightness = 0.99) {
    this._h = hue % 1;
    this._s = saturation % 1;
    this._v = brightness % 1;
    this._map = Object.create(null);
  }
  get(id: any) { return this._map[id] || (this._map[id] = this.generate()); }
  generate() {
    const rgba = HSVtoRGBA(this._h, this._s, this._v);
    this._h = (this._h + goldenRatioConjugate) % 1;
    return rgba;
  }
  palette(start = 0, stop = 0, swizzle = true) {
    const swap = ([r, g, b, a]: number[]) => [b, g, r, a];
    const idxs = Array.from({ length: stop - start }, (_, i) => i);
    const palette = idxs.map((i) => {
      return swizzle ? swap(this.get(i)) : this.get(i);
    }).flat();
    return Series.new(new Uint8Array(palette)).view(new Uint32);
  }
}

// # HSV values in [0..1]
// # returns [r, g, b] values from 0 to 255
function HSVtoRGBA(h: number, s: number, v: number) {
  let r = 0;
  let g = 0;
  let b = 0;
  let i = Math.floor(h * 6);
  let f = h * 6 - i;
  let p = v * (1 - s);
  let q = v * (1 - f * s);
  let t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    case 5: r = v, g = p, b = q; break;
  }
  return [
    Math.round(r * 255),  // r
    Math.round(g * 255),  // g
    Math.round(b * 255),  // b
    255,                  // a
  ] as [number, number, number, number];
}
