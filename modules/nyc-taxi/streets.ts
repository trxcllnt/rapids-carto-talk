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

import { RapidsJSDOM } from '@rapidsai/jsdom';

const dom = new RapidsJSDOM({
  glfwOptions: { width: 1920, height: 1080 },
});

dom.window.addEventListener('close', () => process.exit(0), { once: true });

Object.assign(dom, {
  loaded: dom.loaded.then(() => dom.window.evalFn(async () => await import('./src/render-streets'))),
});
